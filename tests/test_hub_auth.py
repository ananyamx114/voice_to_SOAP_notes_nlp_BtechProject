import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from voice_to_soap.config import huggingface_token


class HubAuthenticationTests(unittest.TestCase):
    def test_explicit_token_overrides_cached_login(self):
        cached = Mock(return_value="test-cached-token")
        with patch.dict(os.environ, {"HF_TOKEN": "test-explicit-token"}, clear=True):
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(get_token=cached)}):
                self.assertEqual(huggingface_token(), "test-explicit-token")
        cached.assert_not_called()

    def test_configured_environment_name_is_respected(self):
        with patch.dict(os.environ, {"CUSTOM_HF_TOKEN": "test-custom-token"}, clear=True):
            self.assertEqual(huggingface_token("CUSTOM_HF_TOKEN"), "test-custom-token")

    def test_cli_cached_login_is_used_without_environment_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(
                get_token=lambda: "test-cached-token"
            )}):
                self.assertEqual(huggingface_token(), "test-cached-token")

    def test_missing_optional_hub_package_is_allowed(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                self.assertIsNone(huggingface_token())
