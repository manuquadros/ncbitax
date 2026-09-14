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

    index_file = tmp_path / "bacteria_name_index.pickle"
    index = {"escherichiacoli": ("Escherichia coli", 562)}
    ncbitax.save_index(index=index, path=index_file)

    assert ncbitax.get_index(index_file) == index

    def normalize(name: str) -> str:
        return name.lower() + "zzz"

    monkeypatch.setattr(ncbitax, "normalize", normalize)

    assert ncbitax.get_index(index_file) == {}
