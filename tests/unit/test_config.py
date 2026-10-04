from __future__ import annotations

from pathlib import Path

import pytest

from meldr import secrets
from meldr.config import MeldrConfig
from meldr.core.errors import ConfigError, SecretReferenceError
from meldr.secrets import SecretRef


def test_literal_secret_rejected() -> None:
    with pytest.raises(ConfigError, match="instead of a literal secret"):
        MeldrConfig.from_dict(
            {"llm": {"providers": {"c": {"type": "anthropic", "api_key": "sk-abc"}}}}
        )


def test_secret_default_not_allowed() -> None:
    with pytest.raises(ConfigError, match="secret fields cannot have defaults"):
        MeldrConfig.from_dict(
            {"llm": {"providers": {"c": {"type": "anthropic", "api_key": "${env:K:-x}"}}}}
        )


def test_unknown_key_has_path() -> None:
    with pytest.raises(ConfigError, match=r"mapping\.auto_aprove_threshold: unknown key"):
        MeldrConfig.from_dict({"mapping": {"auto_aprove_threshold": 0.5}})


def test_env_resolution_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MeldrConfig.from_dict(
        {"llm": {"providers": {"c": {"type": "anthropic", "api_key": "${env:HZ_TEST_KEY}"}}}}
    )
    ref = cfg.llm.providers["c"].api_key
    assert isinstance(ref, SecretRef)
    with pytest.raises(SecretReferenceError, match="HZ_TEST_KEY"):
        ref.reveal()
    monkeypatch.setenv("HZ_TEST_KEY", "value")
    assert ref.reveal() == "value"
    assert "value" not in repr(ref) and "value" not in cfg.model_dump_json()


def test_non_secret_interpolation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HZ_ROOT", str(tmp_path))
    cfg = MeldrConfig.from_dict(
        {
            "storage": {"root": "${env:HZ_ROOT}/state"},
            "embeddings": {"model": "${env:HZ_MISSING:-mini}"},
        }
    )
    assert cfg.storage.root == f"{tmp_path}/state"
    assert cfg.embeddings.model == "mini"


def test_remote_paths_are_not_rebased(tmp_path: Path) -> None:
    cfg = MeldrConfig.from_dict(
        {
            "sources": {"remote": {"connector": "files", "path": "s3://bucket/data"}},
            "sinks": {"tables": {"type": "parquet", "path": "s3://bucket/output"}},
        },
        base_dir=tmp_path,
    )
    assert cfg.sources["remote"].path == "s3://bucket/data"
    assert cfg.sinks.tables.path == "s3://bucket/output"


def test_vault_stub_and_custom_resolver() -> None:
    ref = SecretRef("${vault:secret/data/erp#url}")
    with pytest.raises(SecretReferenceError, match="register"):
        ref.reveal()
    secrets.register("vault", lambda body: f"resolved:{body}")
    try:
        assert ref.reveal() == "resolved:secret/data/erp#url"
    finally:
        secrets.register("vault", secrets._vault_stub)
    with pytest.raises(SecretReferenceError, match="Unknown secret scheme"):
        secrets.resolve("${nope:x}")


def test_load_plan_example(tmp_path: Path) -> None:
    plan = Path("docs/planning/meldr/implementation_plan.md")
    if not plan.exists():
        pytest.skip("docs/planning is local only and absent from a clone")
    text = plan.read_text()
    block = text.split("Example `meldr.yaml`:\n\n```yaml\n", 1)[1].split("```", 1)[0]
    # The plan's example leaves ${env:...} unquoted inside a flow mapping, which is invalid YAML.
    block = block.replace("token: ${env:VENDORS_API_TOKEN}}", "token: '${env:VENDORS_API_TOKEN}'}")
    p = tmp_path / "meldr.yaml"
    p.write_text(block)
    cfg = MeldrConfig.load(p)
    assert cfg.sources["project_docs"].data_policy is not None
    assert Path(cfg.storage.root).is_absolute()
    assert cfg.llm.routing.mapping == "claude"


def test_bad_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a list\n")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        MeldrConfig.load(p)
    with pytest.raises(ConfigError, match="Cannot read"):
        MeldrConfig.load(tmp_path / "missing.yaml")
