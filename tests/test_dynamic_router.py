"""Tests for Core.routing.dynamic_router."""

from Core.routing.dynamic_router import DynamicRouter, RoutingRule


class TestDynamicRouter:
    def test_default_routing_simple_summary(self):
        router = DynamicRouter()
        assert router.get_model("simple_summary") == "lfm2.5-thinking:latest"

    def test_default_routing_complex_code(self):
        router = DynamicRouter()
        assert router.get_model("complex_code") == "deepseek-coder-v2:latest"

    def test_default_routing_vision(self):
        router = DynamicRouter()
        assert router.get_model("vision") == "qwen3-vl:latest"

    def test_fallback_on_unknown_category(self):
        router = DynamicRouter(default_model="fallback:latest")
        assert router.get_model("never_heard_of") == "fallback:latest"

    def test_custom_rules_override_defaults(self):
        custom = [RoutingRule("simple_summary", "custom-model:v1")]
        router = DynamicRouter(rules=custom)
        assert router.get_model("simple_summary") == "custom-model:v1"

    def test_add_rule(self):
        router = DynamicRouter()
        router.add_rule(RoutingRule("new_task", "new-model:latest"))
        assert router.get_model("new_task") == "new-model:latest"

    def test_remove_rule_falls_to_default(self):
        router = DynamicRouter(default_model="default:v1")
        router.remove_rule("simple_summary")
        assert router.get_model("simple_summary") == "default:v1"

    def test_list_rules_returns_all(self):
        router = DynamicRouter()
        rules = router.list_rules()
        names = {r.task_category for r in rules}
        assert "simple_summary" in names
        assert "complex_reasoning" in names
