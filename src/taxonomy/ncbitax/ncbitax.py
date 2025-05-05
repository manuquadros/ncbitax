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
from pandas.api.typing import DataFrameGroupBy

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
            self._table = table + ".dmp"
        else:
            self._table = table

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
        self._tar = tarfile.open(taxdump)
        tbl = self._tar.extractfile(self._table)
        self._file = TextIOWrapper(tbl, encoding="utf-8")
        return self

    def __exit__(self, type, value, traceback):
        self._tar.close()
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


@cache
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
    return re.sub(r"[^a-z0-9]", "", name.lower())


def names() -> pd.DataFrame:
    return load_df("names")


def nodes() -> pd.DataFrame:
    return load_df("nodes")


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


@cache
def bacterial_name_index(rank: str) -> NameIndex:
    _cache_paths = {
        "species": ROOTDIR / "resources/bacteria_name_index.pickle",
        "strain": ROOTDIR / "resources/strain_name_index.pickle",
        "genus": ROOTDIR / "resources/genus_name_index.pickle",
    }
    index_cache_path = _cache_paths[rank]
    index = get_index(index_cache_path)

    if index:
        return index
    else:
        bacnodes = nodes().query("division_id == 0 & rank == 'species'")
        bac_genus_nodes = nodes().query("division_id == 0 & rank == 'genus'")
        name_classes = (
            "synonym",
            "scientific name",
            "equivalent name",
            "common name",
        )

        is_bac_id = "(tax_id in @bacnodes['tax_id'].values)"

        if rank == "species":
            _names = names().query(
                f"{is_bac_id} & (name_class in @name_classes)"
            )
            # For species names, we also remove citations before normalizing
            _names["norm"] = _names["name_txt"].apply(
                lambda n: normalize(remove_citations(n))
            )
            desc = "Bacterial species names"
        elif rank == "genus":
            _names = names().query(
                "tax_id in @bac_genus_nodes['tax_id'].values"
            )
            _names["norm"] = _names["name_txt"].apply(
                lambda n: normalize(remove_citations(n))
            )
            desc = "Bacterial genus names"
        elif rank == "strain":
            strain_nodes = nodes().query("division_id == 0 & rank == 'strain'")
            type_material = f"{is_bac_id} & name_class == 'type material'"
            is_strain_id = "tax_id in @strain_nodes['tax_id'].values"
            strain_node_cond = f"{is_strain_id} & name_class in @name_classes"

            _names = names().query(f"({type_material}) | ({strain_node_cond})")
            _names["norm"] = _names["name_txt"].apply(normalize)
            desc = "Bacterial strain names"

        scinames = dict(
            _names.query("name_class == 'scientific name'")[
                ["tax_id", "name_txt"]
            ].values
        )

        index = {
            row.norm: (scinames.get(row.tax_id, row.name_txt), row.tax_id)
            for row in tqdm(
                _names.itertuples(index=False),
                total=len(_names),
                desc=desc,
            )
        }

    save_index(index=index, path=index_cache_path)

    return index


def resolve_tax_id(query: str) -> int | None:
    """Resolve a scientific name or synonym to its tax_id

    :param query: e.g. 'E. coli', 'Staph. aureus'
    :return: tax_id or None
    """
    normed = normalize(query)
    result = (
        bacterial_name_index("species").get(normed)
        or bacterial_name_index("strain").get(normed)
        or bacterial_name_index("genus").get(normed)
    )

    if result:
        return result[1]

    return None


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


@cache
def _nodes_indexed() -> pd.DataFrame:
    return nodes().set_index("tax_id")


@cache
def _names_by_tax_id() -> DataFrameGroupBy:
    return names().groupby("tax_id")


@cache
def decompose_name(query: str) -> DecomposedName | None:
    """Return the species name and the strain identifier from `query`."""
    if not query:
        return None

    node = get_node(query)

    if node is None:
        query_parts = query.split()[:-1]
        if not query_parts:
            return None
        if query_parts[-1] == "sp.":
            node = get_node(query_parts[0])
            if node is not None and node["rank"] == "genus":
                return DecomposedName(species=query, strain=None)

        return decompose_name(" ".join(query_parts))

    if node["rank"] not in ("strain", "species"):
        return None

    nodes_indexed = _nodes_indexed()  # cached
    names_by_taxid = _names_by_tax_id()  # cached

    def get_name_txt(tax_id: int, name_class: str) -> str | None:
        group = names_by_taxid.get_group(tax_id)
        match = group[group["name_class"] == name_class]
        return match["name_txt"].iloc[0] if not match.empty else None

    current_id = node["tax_id"]
    # Check if `node` is a type strain

    try:
        matches = names_by_taxid.get_group(current_id)
        exact_match = matches[matches["name_txt"] == query]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            if row["name_class"] == "type material":
                return DecomposedName(species=None, strain=query)
            elif node["rank"] == "species":
                return DecomposedName(species=query, strain=None)
    except IndexError:
        # Get the lineage by walking up the taxonomy tree until we hit species rank
        pass

    while True:
        try:
            current_node = nodes_indexed.loc[current_id]
        except KeyError:
            return None

        if current_node["rank"] == "species":
            species_name = get_name_txt(current_id, "scientific name")
            strain_name = get_name_txt(node["tax_id"], "scientific name")

            if species_name and strain_name:
                strain_clean = (
                    strain_name.replace(species_name, "").strip() or None
                )
                return DecomposedName(species=species_name, strain=strain_clean)

            return None

        current_id = current_node["parent_tax_id"]
        if current_id == 1:  # Hit root without finding species
            return DecomposedName(species=None, strain=None)
