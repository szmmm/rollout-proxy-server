"""Offline regression checks for DDB compaction and transient transport recovery."""
import asyncio
import sys
import tempfile
from pathlib import Path

import httpx

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from offline_common import ScriptedProvider
from proxyserver.server import LLMProxyServer, make_local_completion_handler


async def main():
    provider = ScriptedProvider()
    with tempfile.TemporaryDirectory() as temp:
        proxy = LLMProxyServer(
            api_key="test", save_rollout_sessions=False, error_log_dir=temp,
            completion_handler=make_local_completion_handler(provider),
            on_session_deleted=provider.release_session,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app), base_url="http://test") as http:
            async def post(agent, messages, **extra):
                return await http.post("/v1/chat/completions", headers={"Authorization": f"Bearer test-session-{agent}"}, json={"model": "Qwen3.5-9B", "messages": messages, "max_tokens": 32, **extra})

            opener = [{"role": "user", "content": "build"}]
            ordinary = await post("ddt_agent1", opener)
            assert ordinary.status_code == 200 and ordinary.headers["x-request-id"]
            transcript = opener + [{"role": "assistant", "content": "past sampled history"}, {"role": "user", "content": "Summarize the transcript"}]
            rejected = await post("unrelated", transcript)
            assert rejected.status_code == 400  # strict policy histories stay protected
            summary = await post("compaction1", transcript)
            assert summary.status_code == 200
            replay = await post("compaction1", transcript)
            assert replay.status_code == 200
            # Recorder implementations expose records through the endpoint.
            record = (await http.get("/sessions/session")).json()
            compact = [turn for turn in record["turns"] if turn["agent_id"] == "compaction1"]
            assert len(compact) == 1
            assert all(msg["role"] != "assistant" for msg in compact[0]["request_messages"])
            assert "past sampled history" in str(compact[0]["request_messages"])
            original_handler = proxy._completion_handler
            observed_bodies = {}

            async def inspect_body(proxy_arg, session, agent, model, messages, body, streaming):
                observed_bodies[agent] = body
                return await original_handler(proxy_arg, session, agent, model, messages, body, streaming)

            proxy._completion_handler = inspect_body
            inherited_tools = [{"type": "function", "function": {"name": "bash"}}]
            summarized = await post("compaction2", transcript, tools=inherited_tools,
                tool_choice={"type": "function", "function": {"name": "bash"}},
                parallel_tool_calls=True)
            assert summarized.status_code == 200
            assert all(key not in observed_bodies["compaction2"] for key in ("tools", "tool_choice", "parallel_tool_calls"))
            ordinary_tools = await post("ddt_agentWithTools", opener, tools=inherited_tools)
            assert ordinary_tools.status_code == 200
            assert observed_bodies["ddt_agentWithTools"]["tools"] == inherited_tools

            call_engine = provider._call_engine
            attempts = 0

            async def flaky(*args):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise httpx.ConnectError("temporary connection failure")
                return await call_engine(*args)

            provider._call_engine = flaky
            recovered = await post("ddt_agent2", opener)
            assert recovered.status_code == 200 and attempts == 2
            record = (await http.get("/sessions/session")).json()
            assert len([t for t in record["turns"] if t["agent_id"] == "ddt_agent2"]) == 1

            attempts = 0

            async def invalid(*args):
                nonlocal attempts
                attempts += 1
                response = httpx.Response(400, request=httpx.Request("POST", "http://engine"))
                raise httpx.HTTPStatusError("bad request", request=response.request, response=response)

            provider._call_engine = invalid
            rejected = await post("ddt_agent3", opener)
            assert rejected.status_code == 400 and attempts == 1
        await proxy.stop()
    print("DDB reliability: compaction isolation, strict history, replay, transport recovery, and non-retryable errors passed")


if __name__ == "__main__":
    asyncio.run(main())
