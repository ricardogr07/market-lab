from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest

from marketlab import __version__
from marketlab._version import FALLBACK_VERSION, get_version
from marketlab.resources.templates import (
    CONFIG_TEMPLATE_NAMES,
    get_config_template_text,
    iter_config_template_names,
    write_config_template,
)


def test_get_version_uses_fallback_when_distribution_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_package_not_found(_: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, 'version', raise_package_not_found)

    assert get_version() == FALLBACK_VERSION


def test_public_package_version_matches_version_helper() -> None:
    assert __version__ == get_version()


def test_template_registry_exposes_expected_names() -> None:
    assert CONFIG_TEMPLATE_NAMES == (
        'weekly_rank',
        'weekly_rank_smoke',
        'phase5_allocation_equal',
        'phase5_allocation_group',
        'phase5_ranking_default',
        'phase5_ranking_capped',
        'phase5_mean_variance',
        'phase5_risk_parity',
        'phase5_black_litterman',
    )
    assert iter_config_template_names() == CONFIG_TEMPLATE_NAMES


@pytest.mark.parametrize(
    ('template_name', 'repo_config_path'),
    [
        ('weekly_rank', Path('configs/experiment.weekly_rank.yaml')),
        ('weekly_rank_smoke', Path('configs/experiment.weekly_rank.smoke.yaml')),
        ('phase5_allocation_equal', Path('configs/experiment.phase5.allocation_equal.yaml')),
        ('phase5_allocation_group', Path('configs/experiment.phase5.allocation_group.yaml')),
        ('phase5_ranking_default', Path('configs/experiment.phase5.ranking_default.yaml')),
        ('phase5_ranking_capped', Path('configs/experiment.phase5.ranking_capped.yaml')),
        ('phase5_mean_variance', Path('configs/experiment.phase5.mean_variance.yaml')),
        ('phase5_risk_parity', Path('configs/experiment.phase5.risk_parity.yaml')),
        ('phase5_black_litterman', Path('configs/experiment.phase5.black_litterman.yaml')),
    ],
)
def test_packaged_templates_match_repo_config_sources(
    template_name: str,
    repo_config_path: Path,
) -> None:
    assert get_config_template_text(template_name).rstrip('\n') == repo_config_path.read_text(
        encoding='utf-8'
    ).rstrip('\n')


def test_write_config_template_resolves_relative_output_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    written_path = write_config_template('weekly_rank', Path('nested') / 'template.yaml')

    assert written_path == (tmp_path / 'nested' / 'template.yaml').resolve()
    assert written_path.read_text(encoding='utf-8').startswith('experiment_name: weekly_rank_v1')


def test_write_config_template_supports_phase5_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    written_path = write_config_template(
        'phase5_black_litterman',
        Path('nested') / 'phase5_black_litterman.yaml',
    )

    assert written_path == (tmp_path / 'nested' / 'phase5_black_litterman.yaml').resolve()
    assert written_path.read_text(encoding='utf-8').startswith(
        'experiment_name: phase5_black_litterman'
    )
