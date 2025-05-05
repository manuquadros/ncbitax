import pandas as pd
from .ncbitax import (
    decompose_name,
    resolve_tax_id,
    is_bacteria,
    is_bacterial_strain,
    DecomposedName,
)

__all__ = [
    "decompose_name",
    "resolve_tax_id",
    "is_bacteria",
    "is_bacterial_strain",
    "DecomposedName",
]

pd.options.mode.copy_on_write = True
