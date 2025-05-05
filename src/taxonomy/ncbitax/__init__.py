import pandas as pd

pd.options.mode.copy_on_write = True

from .ncbitax import (
    bacterial_name_index,
    decompose_name,
    DecomposedName,
    is_bacteria,
    is_bacterial_strain,
    names,
    nodes,
    normalize,
    remove_citations,  # temp
    resolve_tax_id,
)

__all__ = [
    "bacterial_name_index",
    "decompose_name",
    "DecomposedName",
    "is_bacterial_strain",
    "is_bacteria",
    "names",
    "nodes",
    "normalize",
    "remove_citations",  # temp
    "resolve_tax_id",
]
