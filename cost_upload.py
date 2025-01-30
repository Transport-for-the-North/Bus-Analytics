# -*- coding: utf-8 -*-
"""Script to upload the cost metrics CSV to the PostgreSQL database."""

##### IMPORTS #####

import datetime
import logging
import pathlib
import re

import caf.toolkit as ctk
import pandas as pd
import pydantic
from bodse import database
from otp4gb import routing
from pydantic import dataclasses

import scheduler

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_CONFIG_FILE = pathlib.Path(__file__).with_suffix(".yml")
_TIME_PERIODS = ("AM", "IP", "PM", "OP")

##### CLASSES & FUNCTIONS #####


class _UploadConfig(ctk.BaseConfig):
    """Config file for script uploading cost metrics to database."""

    metrics_path: pydantic.FilePath
    zone_system_id: int
    timetable_id: int
    database_parameters: database.DatabaseConfig


@dataclasses.dataclass
class OTPParameters:
    """Parameters for an OTP cost metrics file."""

    time_period: str
    modes: list[routing.Mode]
    date: datetime.date

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "OTPParameters":
        """Extract cost metrics information from file path."""
        time_period = path.parent.name.strip().upper()
        if time_period not in _TIME_PERIODS:
            raise ValueError(
                f"Invalid time period '{time_period}' should be one of {_TIME_PERIODS}"
            )

        matched = re.match(r"^(\w+)_costs_(\d{8})T\d{4}-metrics$", path.stem, re.I)
        if matched is None:
            raise ValueError(f"Invalid filename '{path.stem}'")

        modes = [routing.Mode(i) for i in matched.group(1).split("_")]
        date = datetime.date.fromisoformat(matched.group(2))

        return cls(time_period=time_period, modes=modes, date=date)


def upload_cost_metrics_file(
    pg_database: database.Database,
    metrics_path: pathlib.Path,
    *,
    run_id: int,
    timetable_id: int,
    zone_system_id: int,
):
    """Read cost metrics CSV and upload to the database."""
    LOG.info('Reading and uploading cost metrics file: "%s"', metrics_path)
    df: pd.DataFrame = pd.read_csv(metrics_path)

    # Update column names
    df.drop(columns=scheduler._OTP_COST_METRICS_DROP_COLUMNS, inplace=True)
    df.rename(columns=scheduler._OTP_COST_METRICS_RENAME_COLUMNS, inplace=True)

    # Add run metadata ID and zone system columns
    df["run_id"] = run_id
    df["timetable_id"] = timetable_id
    df["zone_type_id"] = zone_system_id
    df.columns = [scheduler._camel_to_snake_case(i) for i in df.columns]

    LOG.debug("Uploading %s to database", metrics_path.name)
    with pg_database.engine.connect() as conn:
        df.to_sql(
            "cost_metrics",
            conn,
            schema="bus_data",
            index=False,
            if_exists="append",
        )


def main() -> None:
    """Upload cost metrics CSV and metadata to database."""
    parameters = _UploadConfig.load_yaml(_CONFIG_FILE)

    log_file = _CONFIG_FILE.with_suffix(".log")
    details = ctk.ToolDetails(_CONFIG_FILE.stem, "0.1.0")

    with ctk.LogHelper(_CONFIG_FILE.stem, details, log_file=log_file):
        LOG.info("Connecting to database")
        pg_database = database.Database(parameters.database_parameters)

        metrics_info = OTPParameters.from_path(parameters.metrics_path)

        run_id = pg_database.insert_run_metadata(
            database.ModelName.OTP4GB,
            datetime.datetime.now(),
            {},
            True,
            output=f'Cost metrics upload: "{parameters.metrics_path}"',
            time_period=metrics_info.time_period,
            mode="_".join(i.name for i in metrics_info.modes),
            modelled_date=metrics_info.date,
        )

        upload_cost_metrics_file(
            pg_database,
            parameters.metrics_path,
            run_id=run_id,
            timetable_id=parameters.timetable_id,
            zone_system_id=parameters.zone_system_id,
        )


##### MAIN #####
if __name__ == "__main__":
    main()
