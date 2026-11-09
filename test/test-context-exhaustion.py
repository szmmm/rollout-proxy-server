import json

import pytest

from proxyserver.engines import ContextBudgetExhausted
from proxyserver.rollout_provider import _clamp_to_context_window
from proxyserver.server import _context_budget_exhausted_response


def test_filled_context_raises_typed_exhaustion():
    with pytest.raises(ContextBudgetExhausted):
        _clamp_to_context_window({}, 100, 100, "http://engine", "session-1")


def test_context_exhaustion_is_an_openai_style_400():
    response = _context_budget_exhausted_response(
        ContextBudgetExhausted("context length exceeded")
    )
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["type"] == "invalid_request_error"
