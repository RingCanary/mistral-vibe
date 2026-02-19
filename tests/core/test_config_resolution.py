from __future__ import annotations

from pathlib import Path

import pytest

from vibe.core.paths.config_paths import CONFIG_FILE
from vibe.core.paths.global_paths import GLOBAL_CONFIG_FILE, VIBE_HOME
from vibe.core.config import (
    Backend,
    ModelConfig,
    OAuthConfig,
    ProviderAuthConfig,
    ProviderAuthType,
    ProviderConfig,
    VibeConfig,
    WrongBackendError,
)
from vibe.core.trusted_folders import trusted_folders_manager


class TestResolveConfigFile:
    def test_resolves_local_config_when_exists_and_folder_is_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_config_dir = tmp_path / ".vibe"
        local_config_dir.mkdir()
        local_config = local_config_dir / "config.toml"
        local_config.write_text('active_model = "test"', encoding="utf-8")

        monkeypatch.setattr(trusted_folders_manager, "is_trusted", lambda _: True)

        assert CONFIG_FILE.path == local_config
        assert CONFIG_FILE.path.is_file()
        assert CONFIG_FILE.path.read_text(encoding="utf-8") == 'active_model = "test"'

    def test_resolves_local_config_when_exists_and_folder_is_not_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_config_dir = tmp_path / ".vibe"
        local_config_dir.mkdir()
        local_config = local_config_dir / "config.toml"
        local_config.write_text('active_model = "test"', encoding="utf-8")

        assert CONFIG_FILE.path == GLOBAL_CONFIG_FILE.path

    def test_falls_back_to_global_config_when_local_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Ensure no local config exists
        assert not (tmp_path / ".vibe" / "config.toml").exists()

        assert CONFIG_FILE.path == GLOBAL_CONFIG_FILE.path

    def test_respects_vibe_home_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert VIBE_HOME.path != tmp_path
        monkeypatch.setenv("VIBE_HOME", str(tmp_path))
        assert VIBE_HOME.path == tmp_path


@pytest.mark.parametrize(
    "active_alias,provider_name,provider_api_key_env_var",
    [
        ("openai-alias", "openai", "OPENAI_API_KEY"),
        ("zai-alias", "zai", "ZAI_API_KEY"),
    ],
)
def test_get_active_model_and_provider_for_openai_and_zai(
    active_alias: str,
    provider_name: str,
    provider_api_key_env_var: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ZAI_API_KEY", "zai-test-key")

    config = VibeConfig(
        active_model=active_alias,
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                backend=Backend.GENERIC,
                api_key_env_var="OPENAI_API_KEY",
            ),
            ProviderConfig(
                name="zai",
                api_base="https://api.z.ai/api/paas/v4",
                backend=Backend.GENERIC,
                api_key_env_var="ZAI_API_KEY",
            ),
        ],
        models=[
            ModelConfig(
                name="gpt-4o",
                provider="openai",
                alias="openai-alias",
            ),
            ModelConfig(
                name="zai-model",
                provider="zai",
                alias="zai-alias",
            ),
        ],
        enable_auto_update=False,
    )

    active_model = config.get_active_model()
    provider = config.get_provider_for_model(active_model)

    assert active_model.alias == active_alias
    assert provider.name == provider_name
    assert provider.api_key_env_var == provider_api_key_env_var
    assert provider.backend == Backend.GENERIC


def test_oauth_provider_does_not_require_api_key_env() -> None:
    config = VibeConfig(
        active_model="openai-oauth",
        providers=[
            ProviderConfig(
                name="openai",
                api_base="https://api.openai.com/v1",
                api_key_env_var="",
                backend=Backend.GENERIC,
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(
                        client_id="client-id",
                        token_endpoint="https://auth0.openai.com/oauth/token",
                        device_authorization_endpoint="https://auth0.openai.com/oauth/device/code",
                        scopes=["openid", "offline_access"],
                    ),
                ),
            )
        ],
        models=[
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                alias="openai-oauth",
            )
        ],
        enable_auto_update=False,
    )

    active_model = config.get_active_model()
    provider = config.get_provider_for_model(active_model)

    assert provider.auth.type == ProviderAuthType.OAUTH


def test_openai_chatgpt_provider_requires_openai_chatgpt_backend() -> None:
    config = VibeConfig(
        active_model="codex",
        providers=[
            ProviderConfig(
                name="openai-chatgpt",
                api_base="https://chatgpt.com/backend-api/codex",
                api_style="openai-chatgpt-codex",
                api_key_env_var="",
                backend=Backend.OPENAI_CHATGPT,
                auth=ProviderAuthConfig(
                    type=ProviderAuthType.OAUTH,
                    oauth=OAuthConfig(
                        client_id="client-id",
                        token_endpoint="https://auth.openai.com/oauth/token",
                        device_authorization_endpoint="https://auth.openai.com/api/accounts/deviceauth/usercode",
                        device_poll_endpoint="https://auth.openai.com/api/accounts/deviceauth/token",
                        scopes=["openid", "offline_access"],
                    ),
                ),
            )
        ],
        models=[
            ModelConfig(
                name="gpt-5.3-codex",
                provider="openai-chatgpt",
                alias="codex",
            )
        ],
        enable_auto_update=False,
    )

    active_model = config.get_active_model()
    provider = config.get_provider_for_model(active_model)

    assert active_model.alias == "codex"
    assert provider.backend == Backend.OPENAI_CHATGPT


def test_openai_chatgpt_provider_rejects_generic_backend() -> None:
    with pytest.raises(WrongBackendError):
        VibeConfig(
            active_model="codex",
            providers=[
                ProviderConfig(
                    name="openai-chatgpt",
                    api_base="https://chatgpt.com/backend-api/codex",
                    api_style="openai-chatgpt-codex",
                    api_key_env_var="",
                    backend=Backend.GENERIC,
                    auth=ProviderAuthConfig(type=ProviderAuthType.OAUTH),
                )
            ],
            models=[
                ModelConfig(
                    name="gpt-5.3-codex",
                    provider="openai-chatgpt",
                    alias="codex",
                )
            ],
            enable_auto_update=False,
        )
