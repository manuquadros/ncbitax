import pathlib
import tarfile
from io import TextIOBase, TextIOWrapper, BufferedReader
import pandas as pd
from typing import IO, cast
import sys
import csv
import re

ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent
taxdump = ROOTDIR / "resources/taxdump.tar.gz"

# Increase CSV field size limit to maximum possible to account for long lists
# of tax_id's in rows of the citations table.
# https://stackoverflow.com/a/15063941
field_size_limit = sys.maxsize

while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit = int(field_size_limit / 10)


class NCBIDump(TextIOBase):
    def __init__(self, table: str, sep=","):
        if table[:-4] != ".dmp":
            table = table + ".dmp"

        self._file = TextIOWrapper(
            tarfile.open(taxdump).extractfile(table), encoding="utf-8"
        )
        self._delimiter = r"\t\|\t|\t\|\n"
        self._unquoted_field = re.compile(
            rf"([^\t]*,[^\t]*)(?={self._delimiter})"
        )
        self.sep = sep
        self.unescaped_quote = re.compile(r'(?<!\\)"')

    def preprocess(self, text: str) -> str:
        text = self.unescaped_quote.sub(r"\"", text)
        return self._unquoted_field.sub(r'"\1"', text)

    def readline(self, *args):
        line = self.preprocess(self._file.readline(*args))

        if line:
            return line.replace("\t|\t", self.sep).replace("\t|\n", "\n")
        else:
            return ""

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
    if table[-4:] == ".dmp":
        table = table[:-4]

    filepath = ROOTDIR / f"resources/{table}.parquet.zst"

    try:
        return pd.read_parquet(filepath, engine="pyarrow")
    except FileNotFoundError:
        colinfo = column_info(table)
        with NCBIDump(table) as stream:
            df = pd.read_csv(
                stream,
                engine="python",
                header=None,
                escapechar="\\",
                names=colinfo.keys(),
                dtype=colinfo,
            )

        df.to_parquet(
            filepath, engine="pyarrow", compression="zstd", index=False
        )
        return df


def get_bacteria() -> set[str]:
    """Load the terms that correspond to bacteria in the NCBI taxonomy."""
    # TODO


def is_bacteria(query: str) -> bool:
    """Return True if `query` is a bacteria in the NCBI taxonomy."""
    pass
