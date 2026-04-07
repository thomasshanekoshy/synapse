"""Tests for Core.prompt_registry."""

import pytest

from Core.prompt_registry import PromptRegistry


class TestPromptRegistry:
    def test_register_and_render(self):
        reg = PromptRegistry()
        reg.register("greet", "Hello {name}, you are {role}.")
        result = reg.render("greet", name="Alice", role="admin")
        assert result == "Hello Alice, you are admin."

    def test_version_history(self):
        reg = PromptRegistry()
        reg.register("task", "v1 template")
        reg.register("task", "v2 template")
        assert reg.get("task").version == 2
        assert reg.get("task", version=1).version == 1
        assert len(reg.history("task")) == 2

    def test_get_unknown_raises(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_get_invalid_version_raises(self):
        reg = PromptRegistry()
        reg.register("x", "t")
        with pytest.raises(IndexError):
            reg.get("x", version=99)

    def test_list_prompts(self):
        reg = PromptRegistry()
        reg.register("a", "t1")
        reg.register("b", "t2")
        reg.register("b", "t3")
        listing = reg.list_prompts()
        assert listing == {"a": 1, "b": 2}
