"""Reading CSV / Excel files into DataFrames."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src import utilities as U

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_table(source, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel table from a path or an uploaded file object.

    `source` may be a path, bytes, or a file-like object (e.g. Streamlit's
    UploadedFile). `filename` is needed to detect the type of file objects.
    Raises ValueError with a readable message for unsupported/corrupt files.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext or 'unknown'}'. Please upload a .csv or .xlsx file.")
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        df = pd.read_csv(source) if ext == ".csv" else pd.read_excel(source)
    except Exception as exc:  # corrupt file, wrong encoding, etc.
        raise ValueError(f"Could not read '{Path(name).name}': {exc}") from exc
    if df.empty:
        raise ValueError("The file was read successfully but contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_sample_data() -> pd.DataFrame:
    """Load the bundled synthetic dataset, generating it if it is missing."""
    if not U.SAMPLE_DATA_PATH.exists():
        from src.data_generator import generate_sample_csv
        return generate_sample_csv()
    return pd.read_csv(U.SAMPLE_DATA_PATH, parse_dates=[U.DATE])
