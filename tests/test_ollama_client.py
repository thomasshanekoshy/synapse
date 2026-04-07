"""Tests for Core.models.ollama_client — VRAM management."""

from unittest.mock import patch, MagicMock

from Core.models.ollama_client import OllamaClient


class TestOllamaClientVRAM:
    """Validate the strict 1-model-active VRAM policy."""

    def test_initial_state_is_none(self):
        client = OllamaClient()
        assert client.active_model is None

    def test_ensure_active_sets_model(self):
        client = OllamaClient()
        client.ensure_active("model-a")
        assert client.active_model == "model-a"

    def test_ensure_active_same_model_is_noop(self):
        client = OllamaClient()
        client.ensure_active("model-a")
        with patch.object(client, "unload_model") as mock_unload:
            client.ensure_active("model-a")
            mock_unload.assert_not_called()

    def test_ensure_active_swaps_model(self):
        client = OllamaClient()
        client.ensure_active("model-a")
        with patch.object(client, "unload_model") as mock_unload:
            client.ensure_active("model-b")
            mock_unload.assert_called_once_with("model-a")
        assert client.active_model == "model-b"

    @patch("Core.models.ollama_client.httpx.Client")
    def test_unload_sends_keep_alive_zero(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        client._active_model_id = "model-a"
        client.unload_model("model-a")

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["keep_alive"] == 0
        assert client.active_model is None

    @patch("Core.models.ollama_client.httpx.Client")
    def test_generate_calls_ensure_active(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "hello",
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        client = OllamaClient()
        result = client.generate("test-model", "hi")

        assert client.active_model == "test-model"
        assert result["response"] == "hello"
