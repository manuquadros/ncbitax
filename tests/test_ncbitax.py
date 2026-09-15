import io
import os
import pathlib
import pickle
import tarfile

import pytest

from taxonomy.ncbitax import (
    resolve_tax_id,
    is_bacteria,
    is_bacterial_strain,
    decompose_name,
    DecomposedName,
)
from taxonomy.ncbitax import ncbitax


def test_resolution():
    assert resolve_tax_id("Escherichia coli") is not None
    assert resolve_tax_id("Staphylococcus aureus") is not None
    assert resolve_tax_id("Bacillus subtilis") is not None


def test_is_bacteria_and_strain():
    assert is_bacteria("E. coli")
    assert is_bacteria("Staphylococcus aureus")
    assert not is_bacteria("Homo sapiens")

    assert is_bacterial_strain("Escherichia coli K12")
    assert not is_bacterial_strain("Escherichia coli")
    assert not is_bacterial_strain("Homo sapiens")

    assert is_bacterial_strain("Crocosphaera subtropica ATCC 51142")


def test_decompose_name():
    assert decompose_name("Cronbergia siamensis CCALA 756") == DecomposedName(
        species="Cronbergia siamensis", strain="SAG 11.82"
    )

    assert decompose_name(
        "Raphidiopsis brookii",
    ) == DecomposedName(species="Raphidiopsis brookii", strain=None)

    assert decompose_name("ATCC 51142") == DecomposedName(
        species=None, strain="ATCC 51142"
    )


def test_index_cache_misses_when_the_code_that_built_it_changes(
    tmp_path, monkeypatch
):
    """A cached index is served only while the code that shaped it stands.

    The stored mtime covers names.parquet.zst and nothing else, so without a
    build id a changed normalize() keeps answering off the keys the old one
    produced.
    """
    parquet = tmp_path / "names.parquet.zst"
    parquet.touch()
    monkeypatch.setattr(ncbitax, "NAMES_PARQUET_PATH", parquet)
    monkeypatch.setattr(ncbitax, "taxdump", tmp_path / "taxdump.tar.gz")

    index_file = tmp_path / "bacteria_name_index.pickle"
    index = {"escherichiacoli": ("Escherichia coli", 562)}
    ncbitax.save_index(index=index, path=index_file)

    assert ncbitax.get_index(index_file) == index

    def normalize(name: str) -> str:
        return name.lower() + "zzz"

    monkeypatch.setattr(ncbitax, "normalize", normalize)

    assert ncbitax.get_index(index_file) == {}


def test_index_cache_misses_when_the_parser_of_its_input_changes(
    tmp_path, monkeypatch
):
    """An index is served only while the parser of its frames stands too.

    A changed column_info reshapes names.parquet.zst without moving any mtime
    the pickle stores, so only a build id can catch it.
    """
    parquet = tmp_path / "names.parquet.zst"
    parquet.touch()
    monkeypatch.setattr(ncbitax, "NAMES_PARQUET_PATH", parquet)
    monkeypatch.setattr(ncbitax, "taxdump", tmp_path / "taxdump.tar.gz")

    index_file = tmp_path / "bacteria_name_index.pickle"
    index = {"escherichiacoli": ("Escherichia coli", 562)}
    ncbitax.save_index(index=index, path=index_file)

    assert ncbitax.get_index(index_file) == index

    def column_info(table: str) -> dict[str, str]:
        return {"tax_id": "UInt32"}

    monkeypatch.setattr(ncbitax, "column_info", column_info)

    assert ncbitax.get_index(index_file) == {}


def test_index_cache_misses_when_its_parquet_is_deleted(tmp_path, monkeypatch):
    """An index is dropped with the parquet it was built from.

    The dump stays in place because that is the trap: an index saved against
    parquet and dump both outranks the dump alone, so falling back to the
    dump's mtime would leave the deleted parquet's index standing.
    """
    parquet = tmp_path / "names.parquet.zst"
    parquet.touch()
    dump = tmp_path / "taxdump.tar.gz"
    dump.touch()
    monkeypatch.setattr(ncbitax, "NAMES_PARQUET_PATH", parquet)
    monkeypatch.setattr(ncbitax, "taxdump", dump)

    index_file = tmp_path / "bacteria_name_index.pickle"
    index = {"escherichiacoli": ("Escherichia coli", 562)}
    ncbitax.save_index(index=index, path=index_file)

    assert ncbitax.get_index(index_file) == index

    parquet.unlink()

    assert ncbitax.get_index(index_file) == {}


@pytest.mark.parametrize(
    "corrupt_bytes",
    [
        pytest.param(b"", id="empty"),
        pytest.param(
            pickle.dumps({"mtime": 1.0, "build_id": "x", "data": {}})[:10],
            id="truncated",
        ),
        pytest.param(b"\x00garbage not a pickle at all\xff", id="garbage"),
        pytest.param(pickle.dumps(["not", "a", "dict"]), id="pickled-non-dict"),
    ],
)
def test_get_index_is_a_cache_miss_for_a_corrupt_pickle(
    tmp_path, monkeypatch, corrupt_bytes
):
    """A pickle pickle.load() can't parse, or parses into something other
    than the expected dict, is a miss like any other -- not a crash.

    save_index() writes in place with no .part-then-rename, so an
    interrupted run can leave exactly this behind.
    """
    parquet = tmp_path / "names.parquet.zst"
    parquet.touch()
    monkeypatch.setattr(ncbitax, "NAMES_PARQUET_PATH", parquet)
    monkeypatch.setattr(ncbitax, "taxdump", tmp_path / "taxdump.tar.gz")

    index_file = tmp_path / "bacteria_name_index.pickle"
    index_file.write_bytes(corrupt_bytes)

    assert ncbitax.get_index(index_file) == {}


def test_index_saved_without_a_parquet_never_goes_current(
    tmp_path, monkeypatch
):
    """An index stamped with no parquet under it stays a miss once one is back.

    save_index stamps whatever source_mtime() knows, and with the parquet
    deleted -- under a live process whose frames are already memoized, so
    nothing rewrites it -- that is nothing at all. The pickle outlives the
    deletion, so the stored stamp has to be checked as well as the computed
    one.
    """
    parquet = tmp_path / "names.parquet.zst"
    dump = tmp_path / "taxdump.tar.gz"
    dump.touch()
    monkeypatch.setattr(ncbitax, "NAMES_PARQUET_PATH", parquet)
    monkeypatch.setattr(ncbitax, "taxdump", dump)

    index_file = tmp_path / "bacteria_name_index.pickle"
    index = {"escherichiacoli": ("Escherichia coli", 562)}
    ncbitax.save_index(index=index, path=index_file)

    parquet.touch()

    assert ncbitax.get_index(index_file) == {}


def clear_memos() -> None:
    """Drop every memo that could hold a frame or an index across a rebuild."""
    ncbitax._clear_memos()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A scratch data directory, with the memos cleared either side of it.

    The memos are global and keyed on a table name or a rank alone, so a frame
    or an index read out of `tmp_path` would otherwise outlive the test that
    asked for it.
    """
    monkeypatch.setattr(ncbitax, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ncbitax, "taxdump", tmp_path / "taxdump.tar.gz")
    monkeypatch.setattr(
        ncbitax, "NAMES_PARQUET_PATH", tmp_path / "names.parquet.zst"
    )
    monkeypatch.setenv("NCBITAX_AUTO_DOWNLOAD", "0")

    clear_memos()
    yield tmp_path
    clear_memos()


def write_dump(path: pathlib.Path, **tables: list[tuple[str, ...]]) -> None:
    """Write a taxdump archive holding one `.dmp` member per named table."""
    with tarfile.open(path, "w:gz") as tar:
        for table, rows in tables.items():
            body = "".join("\t|\t".join(row) + "\t|\n" for row in rows).encode()
            info = tarfile.TarInfo(f"{table}.dmp")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


def bacterial_species(tax_id: str) -> tuple[str, ...]:
    """A nodes.dmp row for a species of the bacterial division.

    Only tax_id, rank and division_id decide what an index holds; the eight
    columns after them just have to be there and parse.
    """
    rest = ("0", "11", "1", "0", "1", "0", "0", "")
    return (tax_id, "1", "species", "", "0") + rest


def scientific_name(tax_id: str, name: str) -> tuple[str, ...]:
    """A names.dmp row carrying a scientific name."""
    return (tax_id, name, "", "scientific name")


def make_newest(path: pathlib.Path) -> None:
    """Stamp `path` a second past every file beside it.

    A dump replaced within the same second as the caches it invalidates would
    otherwise read as no newer than they are.
    """
    newest = max(sibling.stat().st_mtime for sibling in path.parent.iterdir())
    os.utime(path, (newest + 1, newest + 1))


def test_parquet_rebuilt_when_the_dump_is_newer(data_dir):
    """A dump replaced under a warm parquet reaches the data."""
    dump = data_dir / "taxdump.tar.gz"
    write_dump(dump, division=[("0", "BCT", "Bacteria", "")])
    assert ncbitax.load_df("division")["division_name"][0] == "Bacteria"

    write_dump(dump, division=[("0", "BCT", "Archaea", "")])
    make_newest(dump)

    clear_memos()
    assert ncbitax.load_df("division")["division_name"][0] == "Archaea"


def test_parquet_rebuilt_when_the_code_that_built_it_changes(
    data_dir, monkeypatch
):
    """A parquet is served only while the parser that shaped it stands."""
    write_dump(
        data_dir / "taxdump.tar.gz", division=[("0", "BCT", "Bacteria", "")]
    )
    assert list(ncbitax.load_df("division").columns) == [
        "division_id",
        "division_cde",
        "division_name",
        "comments",
    ]

    def column_info(table: str) -> dict[str, str]:
        return {
            "id": "UInt8",
            "cde": "string",
            "label": "string",
            "x": "string",
        }

    monkeypatch.setattr(ncbitax, "column_info", column_info)

    clear_memos()
    assert list(ncbitax.load_df("division").columns) == [
        "id",
        "cde",
        "label",
        "x",
    ]


def test_parquet_is_served_when_the_dump_is_gone(data_dir):
    """A data directory holding only parquets must not re-fetch the dump."""
    dump = data_dir / "taxdump.tar.gz"
    write_dump(dump, division=[("0", "BCT", "Bacteria", "")])
    ncbitax.load_df("division")
    dump.unlink()

    clear_memos()
    assert ncbitax.load_df("division")["division_name"][0] == "Bacteria"


def test_name_index_rebuilt_when_the_dump_is_newer(data_dir):
    """A dump replaced under warm caches reaches resolve_tax_id().

    bacterial_name_index reads its pickle before anything reads a parquet, so
    a freshness check confined to load_df never runs on this path.
    """
    dump = data_dir / "taxdump.tar.gz"
    write_dump(
        dump,
        nodes=[bacterial_species("1001")],
        names=[scientific_name("1001", "Escherichia coli")],
    )
    assert resolve_tax_id("Escherichia coli") == 1001

    write_dump(
        dump,
        nodes=[bacterial_species("1001"), bacterial_species("2002")],
        names=[
            scientific_name("1001", "Escherichia coli"),
            scientific_name("2002", "Bacillus subtilis"),
        ],
    )
    make_newest(dump)

    clear_memos()
    assert resolve_tax_id("Bacillus subtilis") == 2002


def test_lookups_agree_after_a_taxon_is_reissued(data_dir):
    """Name index and node frame answer for the same dump, or neither does.

    A tax_id retired and re-issued between dumps is where a half-refreshed
    chain shows: the stale index resolves to the retired id, the fresh frame
    has no row for it, and the answer comes from neither dump.
    """
    dump = data_dir / "taxdump.tar.gz"
    write_dump(
        dump,
        nodes=[bacterial_species("1001")],
        names=[scientific_name("1001", "Escherichia coli")],
    )
    assert is_bacteria("Escherichia coli")

    write_dump(
        dump,
        nodes=[bacterial_species("2002")],
        names=[scientific_name("2002", "Escherichia coli")],
    )
    make_newest(dump)

    clear_memos()
    assert resolve_tax_id("Escherichia coli") == 2002
    assert is_bacteria("Escherichia coli")


class _FakeDownload:
    """A urlopen() response replaying `data` in one chunk."""

    def __init__(self, data: bytes):
        self._data = data
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self) -> "_FakeDownload":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk


def test_download_taxdump_invalidates_a_warm_process(data_dir, monkeypatch):
    """download_taxdump() must not leave a live process split across dumps.

    resolve_tax_id, warmed from an on-disk pickle alone, never touches
    load_df; without download_taxdump clearing both memos together, a lookup
    right after the swap would resolve the retired tax_id through a nodes
    frame already rebuilt from the new dump, an answer neither dump gives.
    """
    dump = data_dir / "taxdump.tar.gz"
    write_dump(
        dump,
        nodes=[bacterial_species("1001")],
        names=[scientific_name("1001", "Escherichia coli")],
    )
    assert resolve_tax_id("Escherichia coli") == 1001

    clear_memos()
    assert resolve_tax_id("Escherichia coli") == 1001  # from the pickle alone

    new_dump = data_dir / "new_taxdump.tar.gz"
    write_dump(
        new_dump,
        nodes=[bacterial_species("2002")],
        names=[scientific_name("2002", "Escherichia coli")],
    )
    payload = new_dump.read_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda url: _FakeDownload(payload)
    )

    ncbitax.download_taxdump()
    make_newest(dump)

    assert resolve_tax_id("Escherichia coli") == 2002
    assert is_bacteria("Escherichia coli")
