# ncbitax

Utilities to navigate NCBI taxonomy dump files.

This package is part of the `taxonomy` namespace:

```python
from taxonomy import ncbitax
```

## Data files

The package needs NCBI's `taxdump.tar.gz` (~70 MB) and builds parquet and
pickle caches from it (~180 MB more). None of this ships in the wheel, so
`ncbitax` resolves a data directory at import time, in this order:

1. `$NCBITAX_DATA_DIR`, if set.
2. `resources/` in the source checkout, when running from a git clone or an
   editable install.
3. `$XDG_CACHE_HOME/ncbitax`, defaulting to `~/.cache/ncbitax`.

If the dump is not in that directory, it is downloaded from
`https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz` the first time it
is needed. Set `NCBITAX_AUTO_DOWNLOAD=0` to turn that off; a missing dump then
raises `TaxdumpNotFoundError` instead.

To fetch the dump ahead of time, for instance in a container build or CI step:

```python
from taxonomy.ncbitax import download_taxdump

download_taxdump()
```

`taxonomy.ncbitax.DATA_DIR` reports the resolved directory.
