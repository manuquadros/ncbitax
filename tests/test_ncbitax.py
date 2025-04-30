from taxonomy.ncbitax import (
    resolve_tax_id,
    is_bacteria,
    is_bacterial_strain,
    decompose_name,
    DecomposedName,
)


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
