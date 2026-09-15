"""Tests for data-directory resolution.

These run without the 69 MB taxonomy dump: they only exercise path
resolution and the error raised when the dump is absent.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

PACKAGE = pathlib.Path(__file__).parent.parent / "src" / "taxonomy" / "ncbitax"


def run(script: str, path: pathlib.Path, env: dict[str, str]) -> str:
    """Run `script` in a subprocess importing ncbitax from `path`."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        # `path` leads PYTHONPATH so its copy of the package wins over any
        # installed one; the rest of the environment is inherited so the
        # subprocess keeps the interpreter's third-party dependencies.
        env=os.environ | {"PYTHONPATH": str(path)} | env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def wheel_layout(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a site-packages tree like the one a wheel install produces.

    The package sits directly under the import root, with no ``pyproject.toml``
    or ``src/`` above it and no ``resources/`` anywhere.
    """
    site_packages = tmp_path / "site-packages"
    shutil.copytree(PACKAGE, site_packages / "taxonomy" / "ncbitax")
    return site_packages


def test_source_checkout_uses_resources_dir(tmp_path):
    """A checkout keeps reading and writing the repo's resources/ dir."""
    data_dir = run(
        "from taxonomy.ncbitax import ncbitax; print(ncbitax.DATA_DIR)",
        path=PACKAGE.parent.parent,
        env={},
    )
    assert pathlib.Path(data_dir) == PACKAGE.parent.parent.parent / "resources"


def test_wheel_install_falls_back_to_cache_dir(tmp_path):
    """An installed wheel must not resolve paths inside site-packages.

    Previously ROOTDIR was __file__.parent x 4, which under a wheel points at
    lib/pythonX.Y -- a directory with no resources/ and no business being
    written to.
    """
    site_packages = wheel_layout(tmp_path)
    cache_home = tmp_path / "cache"

    data_dir = pathlib.Path(
        run(
            "from taxonomy.ncbitax import ncbitax; print(ncbitax.DATA_DIR)",
            path=site_packages,
            env={"XDG_CACHE_HOME": str(cache_home)},
        )
    )

    assert data_dir == cache_home / "ncbitax"
    assert site_packages not in data_dir.parents


def test_env_var_overrides_everything(tmp_path):
    """NCBITAX_DATA_DIR wins over both the checkout and the cache dir."""
    override = tmp_path / "elsewhere"

    for path in (PACKAGE.parent.parent, wheel_layout(tmp_path)):
        data_dir = run(
            "from taxonomy.ncbitax import ncbitax; print(ncbitax.DATA_DIR)",
            path=path,
            env={"NCBITAX_DATA_DIR": str(override)},
        )
        assert pathlib.Path(data_dir) == override


def test_rootdir_is_not_a_public_data_path():
    """The wheel-unsafe checkout-root path must not look like public API.

    A caller that wants the resolved data directory has ``DATA_DIR``; the
    raw ``__file__``-derived root stays private so nothing outside this
    module reaches for a name that is wrong under an installed wheel.
    """
    from taxonomy.ncbitax import ncbitax

    assert not hasattr(ncbitax, "ROOTDIR")
    assert hasattr(ncbitax, "_ROOTDIR")


def test_missing_dump_without_download_raises_clearly(tmp_path):
    """With auto-download off, a missing dump names the file and the way out."""
    empty = tmp_path / "empty"

    message = run(
        """
        from taxonomy.ncbitax import ncbitax

        try:
            ncbitax.taxdump_path()
        except ncbitax.TaxdumpNotFoundError as err:
            print(err)
        """,
        path=wheel_layout(tmp_path),
        env={"NCBITAX_DATA_DIR": str(empty), "NCBITAX_AUTO_DOWNLOAD": "0"},
    )

    assert str(empty / "taxdump.tar.gz") in message
    assert "NCBITAX_DATA_DIR" in message
