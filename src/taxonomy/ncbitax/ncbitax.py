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


def column_info(table: str) -> dict[str, str]:
    """Return names and types for each column in a table of the NCBI dump"""
    match table:
        case "names":
            return {
                "tax_id": "UInt32",
                "name_txt": "string",
                "unique_name": "string",
                "name_class": "category",
            }
        case "nodes":
            return {
                "tax_id": "UInt32",
                "parent_tax_id": "UInt32",
                "rank": "category",
                "embl_code": "category",
                "division_id": "UInt8",
                "inherited_div_flag": "UInt8",
                "genetic_code_id": "UInt64",
                "inherited_gc_flag": "UInt8",
                "mitochondrial_genetic_code": "UInt64",
                "inherited_mgc_flag": "UInt8",
                "genbank_hidden_flag": "UInt8",
                "hidden_subtree_root_flag": "UInt8",
                "comments": "string",
            }
        case "division":
            return {
                "division_id": "UInt8",
                "division_cde": "string",
                "division_name": "string",
                "comments": "string",
            }
        case "gencode":
            return {
                "genetic_code_id": "UInt64",
                "abbreviation": "string",
                "name": "string",
                "cde": "string",
                "starts": "string",
            }
        case "delnodes":
            return {"tax_id": "UInt32"}
        case "merged":
            return {"old_tax_id": "string", "new_tax_id": "string"}
        case "citations":
            return {
                "cit_id": "UInt32",
                "cit_key": "string",
                "medline_id": "UInt32",
                "pubmed_id": "UInt32",
                "url": "string",
                "text": "string",
                "taxid_list": "string",
            }
        case "images":
            return {
                "image_id": "UInt32",
                "image_key": "string",
                "url": "string",
                "license": "category",
                "attribution": "string",
                "source": "category",
                "properties": "string",
                "taxid_list": "UInt32",
            }


def load_df(table: str) -> pd.DataFrame:
    if table[:-4] == ".dmp":
        table = table[:4]

    filepath = ROOTDIR / f"resources/{table}.parquet.zst"

    try:
        return pd.read_parquet(filepath, engine="pyarrow")
    except FileNotFoundError:
        with NCBIDump(table) as table:
        colinfo = column_info(table)
            df = pd.read_csv(
                table,
                sep=r"\t\|\t",
                engine="python",
                header=None,
                names=colinfo.keys(),
                dtype=colinfo,
            )

        df.to_parquet(
            filepath, engine="pyarrow", compression="zstd", index=False
        )
        return df


def names_df() -> pd.DataFrame:
    return load_df("names")
