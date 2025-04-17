import pathlib
import tarfile
from io import TextIOBase, TextIOWrapper, BufferedReader
import pandas as pd
from typing import IO, cast

ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent
taxdump = ROOTDIR / "resources/taxdump.tar.gz"


class NCBIDump(TextIOBase):
    def __init__(self, table: str):
        if table[:-4] != ".dmp":
            table = table + ".dmp"

        self._file = TextIOWrapper(
            tarfile.open(taxdump).extractfile(table), encoding="utf-8"
        )

    def readline(self, *args):
        line = self._file.readline(*args)
        return line[:-3]

    def read(self, *args):
        chunk = self._file.read(*args)
        return chunk.replace("\t|\n", "\n")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()


def colnames(table: str) -> str:
    """Return the column names for each file (table) in the NCBI dump."""
    match table:
        case "names":
            return ("tax_id", "name_txt", "unique_name", "name_class")
        case "nodes":
            return (
                "tax_id",
                "parent_tax_id",
                "rank",
                "embl_code",
                "division_id",
                "inherited_div_flag",
                "genetic_code_id",
                "inherited_gc_flag",
                "mitochondrial_genetic_code",
                "inherited_mgc_flag",
                "genbank_hidden_flag",
                "hidden_subtree_root_flag",
                "comments",
            )
        case "division":
            return ("division_id", "division_cde", "division_name", "comments")
        case "gencode":
            return ("genetic_code_id", "abbreviation", "name", "cde", "starts")
        case "delnodes":
            return ("tax_id",)
        case "merged":
            return ("old_tax_id", "new_tax_id")
        case "citations":
            return (
                "cit_id",
                "cit_key",
                "medline_id",
                "pubmed_id",
                "url",
                "text",
            )
        case "images":
            return (
                "image_id",
                "image_key",
                "url",
                "license",
                "attribution",
                "source",
                "properties",
                "taxid_list",
            )


def load_df(table: str) -> pd.DataFrame:
    if table[:-4] == ".dmp":
        table = table[:4]

    filepath = ROOTDIR / f"resources/{table}.parquet.zst"

    try:
        return pd.read_parquet(filepath, engine="pyarrow")
    except FileNotFoundError:
        with NCBIDump(table) as table:
            df = pd.read_csv(
                table,
                sep=r"\t\|\t",
                engine="python",
                header=None,
                names=colnames(table),
            )

        df.to_parquet(
            filepath, engine="pyarrow", compression="zstd", index=False
        )
        return df


def names_df() -> pd.DataFrame:
    return load_df("names")
