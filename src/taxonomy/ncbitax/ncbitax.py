import csv
import pathlib
import pickle
import re
import sys
import tarfile
from dataclasses import dataclass
from functools import cache
from io import TextIOBase, TextIOWrapper

import pandas as pd

ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent
taxdump = ROOTDIR / "resources/taxdump.tar.gz"
BACTERIA_CACHE_PATH = ROOTDIR / "resources/bacteria_set.pickle"
NAMEINDEX_CACHE_PATH = ROOTDIR / "resources/name_index.pickle"
NODES_PARQUET_PATH = ROOTDIR / "resources/nodes.parquet.zst"
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


def normalize(name: str) -> str:
    """
    Normalize a scientific name for fuzzy matching.

    Removes all non-alphanumeric characters and lowercases the result.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


@cache
def name_index() -> dict[str, tuple[str, int]]:
    """Build mapping from normalized names (and synonyms) to (name, tax_id).
    Cache results.

    Includes:
    - Scientific names
    - Synonyms
    """
    source_mtime = NAMES_PARQUET_PATH.stat().st_mtime

    if NAMEINDEX_CACHE_PATH.exists():
        with open(NAMEINDEX_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
            if cache.get("mtime") == source_mtime:
                return cache["data"]

    names = load_df("names")
    valid_classes = {
        "scientific name",
        "synonym",
        "equivalent name",
        "common name",
        "type_material",
    }

    names = names[names["name_class"].isin(valid_classes)]
    index: dict[str, tuple[str, int]] = {}

    for tax_id, name in zip(names["tax_id"], names["name_txt"]):
        norm = normalize(name)
        index[norm] = (name, tax_id)

    with open(NAMEINDEX_CACHE_PATH, "wb") as f:
        pickle.dump({"mtime": source_mtime, "data": index}, f)

    return index


def resolve_tax_id(query: str) -> int | None:
    """Resolve a scientific name or synonym to its tax_id, with normalization
    Cache results.

    :param query: e.g. 'E. coli', 'Staph. aureus'
    :return: tax_id or None
    """
    return name_index().get(normalize(query), (None, None))[1]


@cache
def get_bacteria() -> set[int]:
    """
    Load the set of tax_id values that represent bacterial species
    in the NCBI taxonomy. Cache to disk, invalidate when source changes.

    :return: A set of tax_ids corresponding to bacteria
    """
    source_mtime = NODES_PARQUET_PATH.stat().st_mtime

    if BACTERIA_CACHE_PATH.exists():
        with open(BACTERIA_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
            if cache.get("mtime") == source_mtime:
                return cache["data"]

    # Load fresh data and rebuild cache
    nodes = load_df("nodes")

    # Build parent → children index
    parent_map: dict[int, list[int]] = {}
    for row in nodes.itertuples():
        parent_map.setdefault(row.parent_tax_id, []).append(row.tax_id)

    # Find all descendants of tax_id 2
    stack = [2]
    bacteria_tax_ids = set()

    while stack:
        current = stack.pop()
        bacteria_tax_ids.add(current)
        children = parent_map.get(current, [])
        stack.extend(children)

    # Filter species in that subtree
    bacteria_species = nodes[
        (nodes["rank"] == "species") & (nodes["tax_id"].isin(bacteria_tax_ids))
    ]

    bacteria_ids = set(bacteria_species["tax_id"])

    with open(BACTERIA_CACHE_PATH, "wb") as f:
        pickle.dump({"mtime": source_mtime, "data": bacteria_ids}, f)

    return bacteria_ids


def get_node(query: str) -> pd.Series | None:
    """Return the `query`'s node in the taxonomy if it exists."""
    tax_id = resolve_tax_id(query)
    nodes = load_df("nodes")
    node_row = nodes[nodes["tax_id"] == tax_id]

    if node_row.empty:
        return None

    return node_row.iloc[0]


def is_bacterial_strain(query: str) -> bool:
    """
    Determine whether the query refers to a bacterial strain or subspecies.

    :param query: Scientific name (possibly fuzzy), e.g., 'E. coli K12'
    :return: True if the resolved tax_id is bacterial and has rank 'strain'
    """
    node = get_node(query)

    if node is None:
        return False

    return node["rank"] == "strain"


@dataclass
class DecomposedName:
    species: str | None
    strain: str | None


def decompose_strain_name(query: str) -> DecomposedName | None:
    """Return the species name and the strain identifier from `query`."""
    node = get_node(query)

    if node is None:
        return node

    # If this isn't a strain, return None, unless it is a species.
    if node["rank"] != "strain":
        if node["rank"] == "species":
            return DecomposedName(species=query, strain=None)
        else:
            return None

    # Get the lineage by walking up the taxonomy tree until we hit species rank
    nodes = load_df("nodes")
    names = load_df("names")

    current_id = node["tax_id"]
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
                strain=strain_name.replace(species_name, "").strip(),
            )

        current_id = current_node["parent_tax_id"]
        if current_id == 1:  # Hit root without finding species
            return DecomposedName(species=None, strain=None)


def is_bacteria(query: str) -> bool:
    """
    Determine whether the given query refers to a bacterial taxon.

    :param query: Scientific name (possibly fuzzy), e.g., 'E. coli'
    :return: True if the query resolves to a bacterial tax_id, else False
    """
    tax_id = resolve_tax_id(query)
    if tax_id is None:
        return False

    return tax_id in get_bacteria()
