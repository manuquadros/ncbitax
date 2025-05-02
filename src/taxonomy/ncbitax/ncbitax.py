import csv
import os
import pathlib
import pickle
import re
import sys
import tarfile
from dataclasses import dataclass
from functools import cache, lru_cache
from loggers import stderr_logger
from io import TextIOBase, TextIOWrapper
from tqdm import tqdm

import pandas as pd

ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent
taxdump = ROOTDIR / "resources/taxdump.tar.gz"
NAMEINDEX_CACHE_PATH = ROOTDIR / "resources/name_index.pickle"
NAMES_PARQUET_PATH = ROOTDIR / "resources/names.parquet.zst"

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
        case _:
            return {}


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


def remove_citations(name: str) -> str:
    year = r"(?:18|19|20)\d\d"
    author = r"(?:[A-Z][a-z]+ )+(?:et al\.? )?"
    citation = re.compile(rf"\(?{author}{year}\)?")
    return citation.sub("", name).strip()


def normalize(name: str) -> str:
    """
    Normalize a scientific name for fuzzy matching.

    Removes all non-alphanumeric characters and lowercases the result.
    """
    name = remove_citations(name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def names() -> pd.DataFrame:
    return load_df("names")


type NameIndex = dict[str, tuple[str, int]]


def source_mtime() -> float:
    NAMES_PARQUET_PATH = ROOTDIR / "resources/names.parquet.zst"

    return NAMES_PARQUET_PATH.stat().st_mtime


def get_index(index_file: os.PathLike) -> NameIndex:
    if index_file.exists():
        with open(index_file, "rb") as f:
            cache = pickle.load(f)
            if cache.get("mtime") >= source_mtime():
                return cache["data"]

    return {}


def save_index(index: NameIndex, path: os.PathLike) -> None:
    with path.open(mode="wb") as f:
        pickle.dump({"mtime": source_mtime(), "data": index}, f)


def bacspecies_nodes() -> pd.DataFrame:
    nodes = load_df("nodes")
    return nodes[(nodes["division_id"] == 0) & (nodes["rank"] == "species")]


def bacstrain_nodes() -> pd.DataFrame:
    nodes = load_df("nodes")
    return nodes[(nodes["division_id"] == 0) & (nodes["rank"] == "strain")]


@cache
def bacteria_name_index() -> NameIndex:
    """Build mapping from normalized bacteria names (and synonyms)
    to (name, tax_id). Cache results.

    Includes:
    - Scientific names
    - Synonyms
    """
    INDEX_CACHE_PATH = ROOTDIR / "resources/bacteria_name_index.pickle"
    index = get_index(INDEX_CACHE_PATH)

    if index:
        return index
    else:
        print("Loading...")
        bacnodes = bacspecies_nodes()
        names = load_df("names")
        names = names[
            names["name_class"].isin(
                ("synonym", "scientific name", "equivalent name", "common name")
            )
        ]
        names = names[names["tax_id"].isin(bacnodes["tax_id"])]
        scinames = dict(
            names[names["name_class"] == "scientific name"][
                ["tax_id", "name_txt"]
            ].values
        )

        names["norm"] = names["name_txt"].apply(normalize)

        index = {
            row.norm: (scinames.get(row.tax_id, row.name_txt), row.tax_id)
            for row in tqdm(
                names.itertuples(index=False),
                total=len(names),
                desc="Bacterial names",
            )
        }

    save_index(index=index, path=INDEX_CACHE_PATH)

    return index




def resolve_tax_id(query: str) -> int | None:
    """Resolve a scientific name or synonym to its tax_id, with normalization
    Cache results.

    :param query: e.g. 'E. coli', 'Staph. aureus'
    :return: tax_id or None
    """
    return name_index().get(normalize(query), (None, None))[1]


@lru_cache
def get_node(query: str) -> pd.Series | None:
    """Return the `query`'s node in the taxonomy if it exists."""
    tax_id = resolve_tax_id(query)
    nodes = load_df("nodes")
    node_row = nodes[nodes["tax_id"] == tax_id]

    if node_row.empty:
        return None

    return node_row.iloc[0]


def get_rank(query: str) -> str | None:
    """Get the taxonomic rank of `query`."""
    node = get_node(query)

    if node is not None:
        return node["rank"]


def is_bacterial_strain(query: str) -> bool:
    """Determine whether the query refers to a bacterial strain.

    :param query: Scientific name (possibly fuzzy), e.g., 'E. coli K12'
    :return: True if the resolved tax_id is bacterial and has rank 'strain'
    """
    return get_rank(query) == "strain"


def is_bacteria(query: str) -> bool:
    """
    Determine whether the given query refers to a bacterial taxon.

    :param query: Scientific name (possibly fuzzy), e.g., 'E. coli'
    :return: True if the query resolves to a bacterial tax_id, else False
    """
    node = get_node(query)

    if node is not None:
        return node["division_id"] == 0

    return False


@dataclass
class DecomposedName:
    species: str | None
    strain: str | None


def decompose_name(query: str) -> DecomposedName | None:
    """Return the species name and the strain identifier from `query`."""
    if not query:
        return None

    node = get_node(query)

    if node is None:
        return decompose_name(" ".join(query.split()[:-1]))

    nodes = load_df("nodes")
    names = load_df("names")

    if node["rank"] not in ("strain", "species"):
        return None

    current_id = node["tax_id"]
    # Check if `node` is a type strain

    try:
        exact_match = names[
            (names["tax_id"] == current_id) & (names["name_txt"] == query)
        ].iloc[0]
        if exact_match["name_class"] == "type material":
            return DecomposedName(species=None, strain=query)
    except IndexError:
        msg = f"No match for {current_id} and {query} in the NCBI Taxonomy."
        stderr_logger().debug(msg)

    # Get the lineage by walking up the taxonomy tree until we hit species rank
    while True:
        current_node = nodes[nodes["tax_id"] == current_id].iloc[0]
        if current_node["rank"] == "species":
            # Found the species, get its name
            species_name = names[
                (names["tax_id"] == current_id)
                & (names["name_class"] == "scientific name")
            ]["name_txt"].iloc[0]

            # Get the strain name
            strain_name = names[
                (names["tax_id"] == node["tax_id"])
                & (names["name_class"] == "scientific name")
            ]["name_txt"].iloc[0]

            return DecomposedName(
                species=species_name,
                strain=strain_name.replace(species_name, "").strip() or None,
            )

        current_id = current_node["parent_tax_id"]
        if current_id == 1:  # Hit root without finding species
            return DecomposedName(species=None, strain=None)
