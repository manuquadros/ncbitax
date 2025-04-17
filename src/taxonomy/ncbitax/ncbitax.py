import pathlib
import tarfile
from io import TextIOBase, TextIOWrapper, BufferedReader
import pandas as pd
from typing import IO, cast

ROOTDIR = pathlib.Path(__file__).parent.parent.parent.parent


class NCBIDump(TextIOBase):
    def __init__(self, file: str | tarfile.ExFileObject):
        if isinstance(file, tarfile.ExFileObject):
            self._file: IO[str] = TextIOWrapper(file, encoding="utf-8")
        else:
            self._file = open(file, "rt")

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


def names_dataframe() -> pd.DataFrame:
    names_stream = cast(
        tarfile.ExFileObject,
        tarfile.open(ROOTDIR / "resources/taxdump.tar.gz").extractfile(
            "names.dmp"
        ),
    )
    filepath = ROOTDIR / "resources/names.parquet.zst"

    try:
        return pd.read_parquet(filepath, engine="pyarrow")
    except FileNotFoundError:
        with NCBIDump(names_stream) as names:
            df = pd.read_csv(
                names,
                sep=r"\t\|\t",
                engine="python",
                header=None,
                names=("tax_id", "name_txt", "unique_name", "name_class"),
            )

        df.to_parquet(
            filepath, engine="pyarrow", compression="zstd", index=False
        )
        return df
