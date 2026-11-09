"""Qwen3 model profile.

Qwen3 uses ChatML, Hermes JSON tool calls, and ``<think>...</think>``
reasoning.  The bundled tokenizer is pinned to the Qwen3-32B snapshot recorded
in ``Qwen3/SOURCE.md``.
"""

from __future__ import annotations

from .base import ModelProfile, ThinkReasoning
from .tool_parser import HermesToolParser


class Qwen3(ModelProfile):
    name = "qwen3"
    tokenizer_dir = "Qwen3"
    generation_prompt = "<|im_start|>assistant\n"
    assistant_turn_end = "<|im_end|>\n"
    tool_call_parser = HermesToolParser
    reasoning = ThinkReasoning()


PROFILE = Qwen3()
