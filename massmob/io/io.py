import gzip
import zipfile
from pathlib import PurePath
import polars as pl

def singlespot_zip_to_points(
    zip_path: str,
    cols_to_keep = ['sptId', 'eventDate', 'latitude', 'longitude', 'countryIso', 'accuracy'],
    cols_to_drop = [
        'consentData', 'consentMedia', 'speed', 'os', 'iabConsentString',
        'systemVersion', 'deviceModel', 'eventId', 'uuid'
    ]
):
    """
    Extract all .gz ndjson files from a ZIP archive, concatenate them, deduplicate,
    and export to a single Parquet file using Polars for maximum performance.

    Args:
        zip_path (str): Path to the input zip archive containing .gz files.
        parquet_path (str): Output path for the final Parquet file.
        cols_to_keep (list): Columns to retain in the dataset.
        cols_to_drop (list): Columns to drop from each file.
        compression (str): Parquet compression algorithm.
        row_group_size (int): Row group size for Parquet export (performance tuning).
    """
    dfs = []  # Will store each DataFrame

    with zipfile.ZipFile(zip_path, "r") as zip_archive:
        # List all .gz files inside the archive
        gz_files = [name for name in zip_archive.namelist() if name.endswith(".gz")]
        n_files = len(gz_files)

        for i, gz_name in enumerate(gz_files):
            print(f"Processing file: {PurePath(gz_name).name} ... ({i + 1}/{n_files})")

            # Open the gzipped ndjson file directly from the zip (no disk extraction)
            with zip_archive.open(gz_name) as zip_fileobj:
                with gzip.open(zip_fileobj, mode="rt", encoding="utf-8") as f:
                    # Read with Polars as NDJSON
                    df = pl.read_ndjson(f)

            # Clean, filter, and deduplicate each chunk
            df = df.drop(cols_to_drop, strict=False)
            df = df.select(cols_to_keep)
            df = df.with_columns([
                # Parse eventDate strings like "2024-06-01 15:12:43 +0200" to datetime (with timezone)
                pl.col('eventDate').str.to_datetime(format="%Y-%m-%d %H:%M:%S %z")
            ])
            df = df.unique()
            dfs.append(df)

            print(f'File {PurePath(gz_name).name} processed. {len(df)} points loaded.')

    # Concatenate all DataFrames into one
    all_df = pl.concat(dfs, rechunk=True)
    print(f"Total points before global deduplication: {len(all_df)}")

    # Global deduplication (across all files)
    all_df = all_df.unique()
    print(f"Total final points (after deduplication): {len(all_df)}")

    # Optional: optimize data types for smaller Parquet size
    pts = all_df.with_columns([
        pl.col("countryIso").cast(pl.Categorical)
        # Add more type casts as desired
    ])
    return pts


def singlespot_zip_to_parquet(
        zip_path: str,
        parquet_path: str,
        compression: str = "zstd",      # "zstd" (recommended), or "snappy", or "gzip"
        row_group_size: int = 100_000,   # Tune for bigger datasets
        **kwargs
    ):
    pts = singlespot_zip_to_points(zip_path, **kwargs)
    # Write a single optimized Parquet file
    pts.write_parquet(
        parquet_path,
        compression=compression,
        row_group_size=row_group_size
    )
    print(f"Parquet export finished: {parquet_path}")
