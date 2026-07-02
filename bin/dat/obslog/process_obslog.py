"""Build processed observation-log tables from raw PFS obslog CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OBSLOG_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DEFAULT_OBSLOG_DIR / "processed"

DEFAULT_RUN_CONFIGS: dict[str, dict[str, Any]] = {
    "march2025": {
        "folder": "obslog_march2025",
        "sequence_prefix": "S25A_march_",
        "dropna_eet_r": False,
    },
    "may2025": {
        "folder": "obslog_may2025",
        "sequence_prefix": "S25A_may_",
        "dropna_eet_r": False,
    },
    "june2025": {
        "folder": "obslog_june2025",
        "sequence_prefix": "S25A_june_",
        "dropna_eet_r": False,
    },
    "sep2025": {
        "folder": "obslog_sep2025",
        "sequence_prefix": "SSP_CO_S25B_Sep_",
        "dropna_eet_r": False,
    },
    "nov2025": {
        "folder": "obslog_nov2025",
        "sequence_prefix": "SSP_CO_",
        "dropna_eet_r": True,
    },
}

DEFAULT_COMBINED_DATE_VISIT_RUNS: dict[str, list[str]] = {
    "S25A": ["march2025", "may2025", "june2025", "sep2025"],
}


def read_observation_log(log_dir: str | Path) -> pd.DataFrame:
    """Read and concatenate raw obslog CSV files from one run directory."""
    log_dir = Path(log_dir)
    csv_files = sorted(log_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No obslog CSV files found in {log_dir}")

    all_data = []
    for file in csv_files:
        print(f"Reading {file}")
        all_data.append(pd.read_csv(file))

    df_all = pd.concat(all_data, ignore_index=True)
    df_all.rename(columns={df_all.columns[0]: "visit_id"}, inplace=True)

    unnamed_columns = [
        column for column in df_all.columns if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        df_all = df_all.drop(columns=unnamed_columns)

    return df_all


def add_utc_date_columns(
    data: pd.DataFrame,
    time_zone: str = "US/Hawaii",
) -> pd.DataFrame:
    """Add UTC timestamp and compact UTC date columns."""
    data = data.copy()
    issued_at = pd.to_datetime(data["issued_at"], errors="coerce")

    if issued_at.dt.tz is None:
        issued_at = issued_at.dt.tz_localize(
            time_zone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        issued_at = issued_at.dt.tz_convert(time_zone)

    data["issued_at"] = issued_at
    data["issued_at_UTC"] = issued_at.dt.tz_convert("UTC")
    data["date"] = data["issued_at_UTC"].dt.strftime("%Y%m%d")
    return data


def filter_observation_log(
    data: pd.DataFrame,
    sequence_prefix: str,
    time_zone: str = "US/Hawaii",
) -> pd.DataFrame:
    filtered_data = data[
        data["sequence_name"].str.startswith(sequence_prefix, na=False)
    ].copy()
    return add_utc_date_columns(filtered_data, time_zone=time_zone)


def processed_paths(
    run_name: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    return {
        "co": output_dir / f"co_{run_name}.csv",
        "date_visit_id": output_dir / f"date_visit_id_{run_name}.csv",
    }


def process_run(
    run_name: str,
    config: dict[str, Any],
    obslog_dir: str | Path = DEFAULT_OBSLOG_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Process one configured run and write its processed CSV outputs."""
    obslog_dir = Path(obslog_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_all = read_observation_log(obslog_dir / config["folder"])
    filtered_data = filter_observation_log(
        df_all,
        sequence_prefix=config["sequence_prefix"],
        time_zone=config.get("time_zone", "US/Hawaii"),
    )

    if config.get("dropna_eet_r", False) and "eet_r" in filtered_data.columns:
        filtered_data = filtered_data.dropna(subset=["eet_r"])

    paths = processed_paths(run_name, output_dir)
    filtered_data.to_csv(paths["co"], index=False)
    filtered_data[["date", "visit_id"]].to_csv(paths["date_visit_id"], index=False)

    print(f"Saved {paths['co']}")
    print(f"Saved {paths['date_visit_id']}")
    return filtered_data


def process_runs(
    run_configs: dict[str, dict[str, Any]],
    obslog_dir: str | Path = DEFAULT_OBSLOG_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    return {
        run_name: process_run(
            run_name,
            config,
            obslog_dir=obslog_dir,
            output_dir=output_dir,
        )
        for run_name, config in run_configs.items()
    }


def read_processed_run(
    run_name: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    return pd.read_csv(processed_paths(run_name, output_dir)["co"])


def combine_date_visit_ids(
    run_names: list[str],
    output_name: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    date_visit_id = []
    for run_name in run_names:
        date_visit_id.append(
            pd.read_csv(processed_paths(run_name, output_dir)["date_visit_id"])
        )

    combined = pd.concat(date_visit_id, ignore_index=True)
    output_file = output_dir / f"date_visit_id_{output_name}.csv"
    combined.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    return combined


def load_config(config_file: str | Path | None) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    if config_file is None:
        return DEFAULT_RUN_CONFIGS, DEFAULT_COMBINED_DATE_VISIT_RUNS

    with Path(config_file).open() as handle:
        config = json.load(handle)

    run_configs = config.get("run_configs", config)
    combined_runs = config.get("combined_date_visit_runs", {})
    return run_configs, combined_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed obslog CSVs from raw obslog directories.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional JSON config. Use either a run_configs object or a top-level "
            "mapping of run names to run configs."
        ),
    )
    parser.add_argument(
        "--obslog-dir",
        default=DEFAULT_OBSLOG_DIR,
        type=Path,
        help="Directory containing raw obslog_* subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        help="Optional subset of configured run names to process.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Do not write combined date_visit_id outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_configs, combined_runs = load_config(args.config)

    if args.runs:
        run_configs = {run_name: run_configs[run_name] for run_name in args.runs}

    process_runs(run_configs, obslog_dir=args.obslog_dir, output_dir=args.output_dir)

    if not args.skip_combined:
        for output_name, run_names in combined_runs.items():
            if all(run_name in run_configs for run_name in run_names):
                combine_date_visit_ids(
                    run_names,
                    output_name=output_name,
                    output_dir=args.output_dir,
                )


if __name__ == "__main__":
    main()
