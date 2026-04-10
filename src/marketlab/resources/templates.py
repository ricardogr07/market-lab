from __future__ import annotations

from importlib import resources
from pathlib import Path

CONFIG_TEMPLATE_NAMES: tuple[str, ...] = (
    "weekly_rank",
    "weekly_rank_smoke",
    "phase5_allocation_equal",
    "phase5_allocation_group",
    "phase5_ranking_default",
    "phase5_ranking_capped",
    "phase5_mean_variance",
    "phase5_risk_parity",
    "phase5_black_litterman",
)
_TEMPLATE_PACKAGE = "marketlab.resources.config_templates"


def iter_config_template_names() -> tuple[str, ...]:
    return CONFIG_TEMPLATE_NAMES


def _template_resource_name(name: str) -> str:
    if name not in CONFIG_TEMPLATE_NAMES:
        expected = ", ".join(CONFIG_TEMPLATE_NAMES)
        raise KeyError(f"Unknown config template: {name!r}. Expected one of: {expected}")
    return f"{name}.yaml"


def get_config_template_text(name: str) -> str:
    template_name = _template_resource_name(name)
    template_path = resources.files(_TEMPLATE_PACKAGE).joinpath(template_name)
    return template_path.read_text(encoding="utf-8")


def write_config_template(name: str, output_path: str | Path, force: bool = False) -> Path:
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(get_config_template_text(name), encoding="utf-8")
    return destination
