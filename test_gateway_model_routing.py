"""Gateway-advertised provider-qualified model IDs (e.g. NIM's "01-ai/yi-large").

``cloud_router.get_provider`` classifies any "vendor/model" ID as OpenRouter.
Models advertised by a configured gateway engine must still be listed in
/v1/models and routed to that engine, not to the direct cloud router.
"""

from __future__ import annotations

import pytest

from openjarvis.engine.multi import MultiEngine
from openjarvis.server.routes import (
    _engine_routes_model,
    _uses_direct_cloud_router,
)


class _FakeEngine:
    def __init__(self, models, engine_id="fake"):
        self._models = list(models)
        self.engine_id = engine_id

    def list_models(self):
        return list(self._models)

    def health(self):
        return True

    def close(self):
        pass


@pytest.mark.parametrize("primary", ["nim", "omniroute", "vllm"])
def test_multi_engine_gateway_model_is_not_direct_cloud(primary):
    multi = MultiEngine(
        [(primary, _FakeEngine(["01-ai/yi-large"])), ("ollama", _FakeEngine([]))]
    )
    assert _engine_routes_model(multi, "01-ai/yi-large")
    assert not _uses_direct_cloud_router(multi, "01-ai/yi-large")


def test_direct_cloud_engine_models_still_use_cloud_router():
    multi = MultiEngine(
        [
            ("nim", _FakeEngine(["01-ai/yi-large"])),
            ("cloud", _FakeEngine(["gpt-4o", "anthropic/claude-x"])),
        ]
    )
    assert _uses_direct_cloud_router(multi, "gpt-4o")
    assert _uses_direct_cloud_router(multi, "anthropic/claude-x")
    assert _uses_direct_cloud_router(multi, "meta/not-advertised")


def test_single_engine_legacy_behaviour():
    assert _uses_direct_cloud_router(_FakeEngine([], "ollama"), "gpt-4o")
    assert not _uses_direct_cloud_router(_FakeEngine([], "litellm"), "openai/x")
    assert not _uses_direct_cloud_router(_FakeEngine([], "nim"), "01-ai/yi-large")
