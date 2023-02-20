#!/usr/bin/env python3

__author__ = "Jeff"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Jeff"]
__email__ = "jiefu.zhu@tu-dresden.de"

import pytest

from ..roc import *
from ..roc.__main__ import read_table


# pytest for read_table()
def test_read_table():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import pandas as pd

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # test csv
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_csv(tmpdir / "test.csv", index=False)
        assert (read_table(tmpdir / "test.csv") == df).all().all()

        # test xlsx
        df.to_excel(tmpdir / "test.xlsx", index=False)
        assert (read_table(tmpdir / "test.xlsx") == df).all().all()