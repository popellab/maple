"""Regression tests for ``maple.extraction.pipeline.make_model``.

The function should construct an ``OpenAIResponsesModel`` whose underlying
``AsyncOpenAI`` client uses fail-fast transport settings (5-minute read
timeout, zero internal retries). openai-python's stock defaults
(``timeout=600`` × ``max_retries=2``) compound a single server-side hang
into ~30 minutes of silent waiting before the ``APITimeoutError``
surfaces; the maple defaults abort in 5 minutes on the first attempt and
let pydantic-ai's structural retry layer take over.
"""

import openai

from maple.extraction.pipeline import (
    DEFAULT_OPENAI_HTTP_TIMEOUT_S,
    DEFAULT_OPENAI_MAX_RETRIES,
    make_model,
)


def _client_of(model) -> openai.AsyncOpenAI:
    """Return the AsyncOpenAI instance the model's provider will use."""
    return model.client  # OpenAIResponsesModel exposes ``.client``


def test_default_timeout_and_max_retries() -> None:
    """The defaults override openai-python's stock 600s / 2 retries."""
    assert DEFAULT_OPENAI_HTTP_TIMEOUT_S == 300.0
    assert DEFAULT_OPENAI_MAX_RETRIES == 0

    model = make_model("gpt-5")
    client = _client_of(model)
    # AsyncOpenAI normalizes the timeout into an httpx.Timeout when set as
    # a float. Either form should report the float we passed.
    timeout = client.timeout
    if hasattr(timeout, "read"):
        assert timeout.read == 300.0
    else:
        assert float(timeout) == 300.0
    assert client.max_retries == 0


def test_overrides_take_effect() -> None:
    """Callers can dial the timeout / max_retries via keyword arguments."""
    model = make_model("gpt-5", http_timeout_s=45.0, openai_max_retries=1)
    client = _client_of(model)
    timeout = client.timeout
    if hasattr(timeout, "read"):
        assert timeout.read == 45.0
    else:
        assert float(timeout) == 45.0
    assert client.max_retries == 1
