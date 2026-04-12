from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = Path(os.environ.get('MARKETLAB_DIST_DIR', ROOT / 'dist')).resolve()
SCRATCH_DIR = Path(os.environ.get('MARKETLAB_PACKAGE_SMOKE_DIR', ROOT / '.package-smoke')).resolve()
EXPECTED_TEMPLATES = (
    ('weekly_rank', 'weekly_rank.yaml'),
    ('weekly_rank_smoke', 'weekly_rank_smoke.yaml'),
    ('phase5_allocation_equal', 'phase5_allocation_equal.yaml'),
    ('phase5_allocation_group', 'phase5_allocation_group.yaml'),
    ('phase5_ranking_default', 'phase5_ranking_default.yaml'),
    ('phase5_ranking_capped', 'phase5_ranking_capped.yaml'),
    ('phase5_mean_variance', 'phase5_mean_variance.yaml'),
    ('phase5_risk_parity', 'phase5_risk_parity.yaml'),
    ('phase5_black_litterman', 'phase5_black_litterman.yaml'),
)


def _find_artifacts() -> tuple[Path, Path]:
    if not DIST_DIR.exists():
        raise RuntimeError(f'Expected built distributions in {DIST_DIR}')

    wheels = sorted(DIST_DIR.glob('marketlab-*.whl'), key=lambda path: path.stat().st_mtime)
    sdists = sorted(DIST_DIR.glob('marketlab-*.tar.gz'), key=lambda path: path.stat().st_mtime)
    if not wheels or not sdists:
        raise RuntimeError(f'Expected wheel and sdist artifacts in {DIST_DIR}')
    return wheels[-1], sdists[-1]


def _open_zip_with_retry(path: Path, attempts: int = 10, delay_seconds: float = 0.2) -> zipfile.ZipFile:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return zipfile.ZipFile(path)
        except PermissionError as error:
            last_error = error
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Unable to open zip archive {path}')


def _open_tar_with_retry(path: Path, attempts: int = 10, delay_seconds: float = 0.2) -> tarfile.TarFile:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return tarfile.open(path, 'r:gz')
        except PermissionError as error:
            last_error = error
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'Unable to open tar archive {path}')


def _assert_wheel_contents(wheel_path: Path) -> None:
    with _open_zip_with_retry(wheel_path) as archive:
        names = set(archive.namelist())

    if not any(name.endswith('dist-info/licenses/LICENSE') for name in names):
        raise RuntimeError('Wheel is missing LICENSE')

    for _, template_name in EXPECTED_TEMPLATES:
        template_path = f'marketlab/resources/config_templates/{template_name}'
        if template_path not in names:
            raise RuntimeError(f'Wheel is missing packaged template {template_path}')


def _assert_sdist_contents(sdist_path: Path) -> None:
    with _open_tar_with_retry(sdist_path) as archive:
        names = set(archive.getnames())

    if not any(name.endswith('/LICENSE') for name in names):
        raise RuntimeError('sdist is missing LICENSE')

    for _, template_name in EXPECTED_TEMPLATES:
        suffix = f'/src/marketlab/resources/config_templates/{template_name}'
        if not any(name.endswith(suffix) for name in names):
            raise RuntimeError(f'sdist is missing packaged template {template_name}')


def _venv_paths(venv_dir: Path) -> tuple[Path, Path, Path]:
    if os.name == 'nt':
        return (
            venv_dir / 'Scripts' / 'python.exe',
            venv_dir / 'Scripts' / 'marketlab.exe',
            venv_dir / 'Scripts' / 'marketlab-mcp.exe',
        )
    return (
        venv_dir / 'bin' / 'python',
        venv_dir / 'bin' / 'marketlab',
        venv_dir / 'bin' / 'marketlab-mcp',
    )


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def _assert_installed_cli(wheel_path: Path, temp_dir: Path, env: dict[str, str]) -> None:
    venv_dir = temp_dir / 'venv'
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_path, marketlab_path, marketlab_mcp_path = _venv_paths(venv_dir)

    wheel_uri = wheel_path.resolve().as_uri()
    _run(
        [str(python_path), '-m', 'pip', 'install', f'marketlab[mcp] @ {wheel_uri}'],
        env=env,
    )

    version_run = _run([str(marketlab_path), '--version'], env=env)
    if not version_run.stdout.strip().startswith('marketlab '):
        raise RuntimeError("Installed CLI --version output was not prefixed with 'marketlab '")

    templates_run = _run([str(marketlab_path), 'list-configs'], env=env)
    template_lines = templates_run.stdout.strip().splitlines()
    if template_lines != [name for name, _ in EXPECTED_TEMPLATES]:
        raise RuntimeError(f'Installed CLI list-configs output was {template_lines}')

    output_path = temp_dir / 'weekly_rank.yaml'
    write_run = _run(
        [
            str(marketlab_path),
            'write-config',
            '--name',
            'weekly_rank',
            '--output',
            str(output_path),
        ],
        env=env,
    )
    if output_path.resolve() != Path(write_run.stdout.strip()):
        raise RuntimeError('Installed CLI write-config did not print the resolved output path')
    if 'experiment_name: weekly_rank_v1' not in output_path.read_text(encoding='utf-8'):
        raise RuntimeError('Installed CLI write-config did not write the expected template')

    scenario_output_path = temp_dir / 'phase5_black_litterman.yaml'
    scenario_write_run = _run(
        [
            str(marketlab_path),
            'write-config',
            '--name',
            'phase5_black_litterman',
            '--output',
            str(scenario_output_path),
        ],
        env=env,
    )
    if scenario_output_path.resolve() != Path(scenario_write_run.stdout.strip()):
        raise RuntimeError('Installed CLI phase5 write-config did not print the resolved output path')
    if 'experiment_name: phase5_black_litterman' not in scenario_output_path.read_text(
        encoding='utf-8'
    ):
        raise RuntimeError('Installed CLI phase5 write-config did not write the expected template')

    mcp_help_run = _run([str(marketlab_mcp_path), '--help'], env=env)
    if '--workspace-root' not in mcp_help_run.stdout or '--artifact-root' not in mcp_help_run.stdout:
        raise RuntimeError('Installed marketlab-mcp --help output did not include the expected flags')


def main() -> int:
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'TMP': str(SCRATCH_DIR), 'TEMP': str(SCRATCH_DIR), 'TMPDIR': str(SCRATCH_DIR)})
    try:
        wheel_path, sdist_path = _find_artifacts()
        _assert_wheel_contents(wheel_path)
        _assert_sdist_contents(sdist_path)
        _assert_installed_cli(wheel_path, SCRATCH_DIR, env)
        print(f'Verified package artifacts: {wheel_path.name} and {sdist_path.name}')
        return 0
    finally:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main())
