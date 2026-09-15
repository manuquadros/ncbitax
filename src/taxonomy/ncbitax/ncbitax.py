import csv
import hashlib
import inspect
import os
import pathlib
import pickle
import re
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import cache, lru_cache
from io import TextIOBase, TextIOWrapper

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.typing import DataFrameGroupBy
from tqdm import tqdm

# Private: only valid for _in_source_checkout()'s own check, and wrong under
# a wheel install. DATA_DIR below is the wheel-safe constant callers want.
_ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"


class TaxdumpNotFoundError(FileNotFoundError):
    """Raised when the NCBI taxonomy dump is neither present nor obtainable."""


def _in_source_checkout() -> bool:
    """Whether the module is being imported from a source tree.

    True for a git checkout and for an editable install, both of which keep
    ``__file__`` inside ``src/taxonomy/ncbitax``. False for a wheel install,
    where ``_ROOTDIR`` lands on an unrelated directory such as
    ``lib/python3.12``.
    """
    return (_ROOTDIR / "pyproject.toml").is_file() and (
        _ROOTDIR / "src" / "taxonomy"
    ).is_dir()


def _resolve_data_dir() -> pathlib.Path:
    """Locate the directory holding the taxonomy dump and its derived caches.

    Resolution order: ``$NCBITAX_DATA_DIR``, then ``resources/`` of the source
    checkout, then a user cache directory. The dump and the parquet/pickle
    caches built from it total well over 200 MB, so they are never shipped in
    the wheel and never written into the installed package.
    """
    if env_dir := os.environ.get("NCBITAX_DATA_DIR"):
        return pathlib.Path(env_dir).expanduser()

    if _in_source_checkout():
        return _ROOTDIR / "resources"

    cache_home = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return pathlib.Path(cache_home).expanduser() / "ncbitax"


DATA_DIR = _resolve_data_dir()
taxdump = DATA_DIR / "taxdump.tar.gz"
NAMES_PARQUET_PATH = DATA_DIR / "names.parquet.zst"


def _auto_download_enabled() -> bool:
    disabled = {"0", "false", "no", "off"}
    return (
        os.environ.get("NCBITAX_AUTO_DOWNLOAD", "1").strip().lower()
        not in disabled
    )


def download_taxdump(dest: pathlib.Path | None = None) -> pathlib.Path:
    """Fetch ``taxdump.tar.gz`` from the NCBI FTP server into `dest`.

    The download goes to a temporary sibling file that is renamed into place
    only once complete, so an interrupted transfer never leaves a truncated
    archive behind for the next run to choke on.
    """
    dest = taxdump if dest is None else dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")

    try:
        with urllib.request.urlopen(TAXDUMP_URL) as response:
            total = int(response.headers.get("Content-Length") or 0)
            with (
                partial.open("wb") as out,
                tqdm(
                    total=total or None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading taxdump",
                ) as progress,
            ):
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    progress.update(len(chunk))
    except (urllib.error.URLError, OSError) as err:
        partial.unlink(missing_ok=True)
        raise TaxdumpNotFoundError(
            f"Could not download the NCBI taxonomy dump from {TAXDUMP_URL}: "
            f"{err}. Download it manually and place it at {dest}, or point "
            f"NCBITAX_DATA_DIR at a directory that already contains it."
        ) from err

    partial.replace(dest)
    return dest


def taxdump_path() -> pathlib.Path:
    """Return the path to ``taxdump.tar.gz``, downloading it on first use."""
    if taxdump.exists():
        return taxdump

    if _auto_download_enabled():
        return download_taxdump(taxdump)

    raise TaxdumpNotFoundError(
        f"No NCBI taxonomy dump at {taxdump}, and automatic download is "
        "disabled by NCBITAX_AUTO_DOWNLOAD. Fetch it from "
        f"{TAXDUMP_URL} and place it there, or set NCBITAX_DATA_DIR to a "
        "directory that already contains it."
    )


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
    def __init__(self, table: str):
        self._table = table + (".dmp" if not table.endswith(".dmp") else "")

        self._delimiter = r"\t\|\t|\t\|\n"
        self._unquoted_field = re.compile(
            rf"([^\t]*,[^\t]*)(?={self._delimiter})"
        )
        self.unescaped_quote = re.compile(r'(?<!\\)"')

    def preprocess(self, text: str) -> str:
        text = self.unescaped_quote.sub(r"\"", text)
        return self._unquoted_field.sub(r'"\1"', text)

    def readline(self, *args):
        line = self.preprocess(self._file.readline(*args))

        if line:
            return line.replace("\t|\t", ",").replace("\t|\n", "\n")
        return ""

    def __enter__(self):
        self._tar = tarfile.open(taxdump_path())
        tbl = self._tar.extractfile(self._table)
        self._file = TextIOWrapper(tbl, encoding="utf-8")
        return self

    def __exit__(self, type_, value, traceback):
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


_PARQUET_BUILD_ID_KEY = b"ncbitax_build_id"


def _parquet_build_id() -> str:
    """Fingerprint of the code that turns a table of the dump into a parquet.

    Hashing the source spares anyone having to remember a version bump, at the
    cost of a rebuild when a comment inside one of these moves -- the cheap
    direction to be wrong in, and the trade the name indexes already make.
    """
    source = "".join(
        inspect.getsource(obj) for obj in (column_info, NCBIDump, load_df)
    )
    return hashlib.sha256(source.encode()).hexdigest()


def _parquet_is_current(filepath: pathlib.Path) -> bool:
    """Whether a cached parquet still answers for the dump and the parser.

    The dump is compared by mtime, since ``download_taxdump`` renames a fresh
    archive into place; a dump that is absent cannot be compared against, and
    the parquet is then the only copy of that data there is. The parser is
    compared through a build id stamped into the parquet's own footer, so one
    written before the stamp existed carries none and is a miss.
    """
    if not filepath.exists():
        return False

    if taxdump.exists() and filepath.stat().st_mtime < taxdump.stat().st_mtime:
        return False

    metadata = pq.read_schema(filepath).metadata or {}
    return metadata.get(_PARQUET_BUILD_ID_KEY) == _parquet_build_id().encode()


@cache
def load_df(table: str) -> pd.DataFrame:
    if table[-4:] == ".dmp":
        table = table[:-4]

    filepath = DATA_DIR / f"{table}.parquet.zst"

    if _parquet_is_current(filepath):
        return pd.read_parquet(filepath, engine="pyarrow")

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

    filepath.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(arrow_table.schema.metadata or {})
    metadata[_PARQUET_BUILD_ID_KEY] = _parquet_build_id().encode()
    pq.write_table(
        arrow_table.replace_schema_metadata(metadata),
        filepath,
        compression="zstd",
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


def source_mtime() -> float | None:
    """Mtime of the newest data a name index is derived from, if any.

    The dump counts even though an index is built out of the parquets: a
    parquet older than the dump is rebuilt from it before the index is, so an
    index keyed on the parquet alone would go on answering for a dump that has
    already been replaced. A dump that is absent cannot be compared against.
    An absent names parquet yields None, not the dump's mtime alone: an index
    saved against parquet and dump both outranks the dump, so deleting the
    parquets by hand would leave that index standing.
    """
    try:
        parquet_mtime = NAMES_PARQUET_PATH.stat().st_mtime
    except FileNotFoundError:
        return None

    if taxdump.exists():
        return max(parquet_mtime, taxdump.stat().st_mtime)

    return parquet_mtime


def _index_build_id() -> str:
    """Fingerprint of the code that decides what a saved index contains.

    Hashing the source means nobody has to remember to bump a version, at the
    cost of a rebuild when a comment inside one of these functions changes --
    the cheap direction to be wrong in. Reading the source needs the ``.py``
    files, which both a checkout and the wheel ship. The parquet build id is
    folded in because an index is built out of the frames that id covers, so a
    reshaped frame has to expire the index too.
    """
    source = "".join(
        inspect.getsource(fn)
        for fn in (normalize, remove_citations, bacterial_name_index)
    )
    return hashlib.sha256((_parquet_build_id() + source).encode()).hexdigest()


def get_index(index_file: pathlib.Path) -> NameIndex:
    if index_file.exists():
        with open(index_file, "rb") as f:
            cache = pickle.load(f)

        data_mtime = source_mtime()
        stored_mtime = cache.get("mtime")
        # The mtime covers the data the index was derived from; the build id
        # covers the code that derived it. A pickle written before the id
        # existed carries none, a pickle saved while the parquet was gone
        # carries no mtime, and a parquet deleted since leaves none to compare
        # against: each is a miss.
        if (
            cache.get("build_id") == _index_build_id()
            and data_mtime is not None
            and stored_mtime is not None
            and stored_mtime >= data_mtime
        ):
            return cache["data"]

    return {}


def save_index(index: NameIndex, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="wb") as f:
        pickle.dump(
            {
                "mtime": source_mtime(),
                "build_id": _index_build_id(),
                "data": index,
            },
            f,
        )


@cache
def bacterial_name_index(rank: str) -> NameIndex:
    _cache_paths = {
        "species": DATA_DIR / "bacteria_name_index.pickle",
        "strain": DATA_DIR / "strain_name_index.pickle",
        "genus": DATA_DIR / "genus_name_index.pickle",
    }
    index_cache_path = _cache_paths[rank]
    index = get_index(index_cache_path)

    if index:
        return index

    # bacnodes, bac_genus_nodes and name_classes are referenced below only as
    # `@name` inside .query() strings, which pandas resolves from this local
    # scope at call time -- invisible to ruff's static unused-variable check.
    bacnodes = nodes().query(  # noqa: F841
        "division_id == 0 & rank == 'species'", engine="python"
    )
    bac_genus_nodes = nodes().query(  # noqa: F841
        "division_id == 0 & rank == 'genus'", engine="python"
    )
    name_classes = (  # noqa: F841
        "synonym",
        "scientific name",
        "equivalent name",
        "common name",
    )

    is_bac_id = "(tax_id in @bacnodes['tax_id'].values)"

    if rank == "species":
        _names = names().query(
            f"{is_bac_id} & (name_class in @name_classes)", engine="python"
        )
        # For species names, we also remove citations before normalizing
        _names["norm"] = _names["name_txt"].apply(
            lambda n: normalize(remove_citations(n))
        )
        desc = "Bacterial species names"
    elif rank == "genus":
        _names = names().query(
            "tax_id in @bac_genus_nodes['tax_id'].values", engine="python"
        )
        _names["norm"] = _names["name_txt"].apply(
            lambda n: normalize(remove_citations(n))
        )
        desc = "Bacterial genus names"
    elif rank == "strain":
        # strain_nodes: same @name-in-query() case as above.
        strain_nodes = nodes().query(  # noqa: F841
            "division_id == 0 & rank == 'strain'", engine="python"
        )
        type_material = f"{is_bac_id} & name_class == 'type material'"
        is_strain_id = "tax_id in @strain_nodes['tax_id'].values"
        strain_node_cond = f"{is_strain_id} & name_class in @name_classes"

        _names = names().query(
            f"({type_material}) | ({strain_node_cond})", engine="python"
        )
        _names["norm"] = _names["name_txt"].apply(normalize)
        desc = "Bacterial strain names"

    scinames = dict(
        _names.query("name_class == 'scientific name'", engine="python")[
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

    return None


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


def get_name_txt(tax_id: int, name_class: str) -> str | None:
    """Get name of `name_class` type for `tax_id."""
    names_by_taxid = _names_by_tax_id()  # cached
    group = names_by_taxid.get_group(tax_id)
    match = group[group["name_class"] == name_class]
    return match["name_txt"].iloc[0] if not match.empty else None


@cache
def decompose_name(query: str) -> DecomposedName | None:
    """Return the species name and the strain identifier from `query`."""
    if not query:
        return None

    node = get_node(query)

    if node is None:
        query_parts = query.split()[:-1]
        if query_parts and query_parts[-1] == "sp.":
            node = get_node(query_parts[0])
            if node is not None and node["rank"] == "genus":
                return DecomposedName(species=query, strain=None)

        return decompose_name(" ".join(query_parts))

    if node["rank"] not in ("strain", "species"):
        return None

    nodes_indexed = _nodes_indexed()  # cached
    names_by_taxid = _names_by_tax_id()  # cached

    current_id = node["tax_id"]
    # Check if `node` is a type strain

    try:
        matches = names_by_taxid.get_group(current_id)
        exact_match = matches[matches["name_txt"] == query]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            if row["name_class"] == "type material":
                return DecomposedName(species=None, strain=query)
            if node["rank"] == "species":
                return DecomposedName(species=query, strain=None)
    except IndexError:
        pass

    # Get the lineage by walking up the taxonomy tree until we hit species rank
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
