# -*- coding: utf-8 -*-
"""
    Dummy example running OTP process
"""

##### IMPORTS #####

import datetime
import logging
import os
import pathlib
import shutil
import sys
from typing import Optional

import caf.toolkit as ctk
import pandas as pd
import pydantic
import sqlalchemy
from bodse import database
from pydantic import dataclasses
from sqlalchemy import orm, sql

# OTP4GB dependency needs to be manually installed and setup from GitHub https://github.com/transport-for-the-north/otp4gb-py/
# Environment variable can be used to define local repo location, although default should suffice
OTP4GB_REPO_FOLDER = os.environ.get("OTP4GB_REPO_FOLDER", "../otp4gb-py")
sys.path.append(OTP4GB_REPO_FOLDER)
# pylint: disable=wrong-import-position
from otp4gb import config, cost, otp, parameters, routing

# pylint: enable=wrong-import-position

##### CONSTANTS #####

LOG = logging.getLogger(__name__)
_CONFIG_FILE = pathlib.Path(__file__).with_suffix(".yml")

_OTP_COST_METRICS_DROP_COLUMNS = [
    "origin",
    "destination",
    "origin_zone_system",
    "destination_zone_system",
]
_OTP_COST_METRICS_RENAME_COLUMNS = {
    "origin_id": "origin_zones_id",
    "destination_id": "destination_zones_id",
}
_OTP_CENTROIDS_RENAME = {
    "id": "zone_id",
    "name": "zone_name",
    "zone_type_id": "zone_system",
}

##### CLASSES & FUNCTIONS #####


@dataclasses.dataclass
class OTPParameters:
    osm_file: str
    time_periods: list[config.TimePeriod]
    modes: list[list[routing.Mode]]
    generalised_cost_factors: cost.GeneralisedCostFactors
    iterinary_aggregation_method: cost.AggregationMethod = cost.AggregationMethod.MEAN
    max_walk_distance: int = pydantic.Field(2500, ge=0)
    number_of_threads: int = pydantic.Field(0, ge=0, le=10)
    crowfly_max_distance: Optional[float] = None
    ruc_lookup: Optional[parameters.RUCLookup] = None


@dataclasses.dataclass
class ZoningSystemParams:
    id: int
    name: str
    extents: config.Bounds


class SchedulerConfig(ctk.BaseConfig):
    output_folder: pydantic.DirectoryPath
    database_parameters: database.DatabaseConfig
    otp_parameters: OTPParameters
    zoning_system_parameters: list[ZoningSystemParams]


@dataclasses.dataclass
class TimetableData:
    id: int
    path: pydantic.FilePath
    date: datetime.date


def next_day(date: datetime.date, weekday: int) -> datetime.date:
    """Get date of next specific weekday."""
    if weekday > 6 or weekday < 0:
        raise ValueError(f"invalid value for weekday: {weekday}")

    days = (weekday - date.weekday() + 7) % 7
    return date + datetime.timedelta(days=days)


def download_timetable_to_assets(
    pg_database: database.Database,
    adjusted: bool = False,
    folder: pathlib.Path = config.ASSET_DIR,
) -> TimetableData:
    """Get GTFS timetable from database and save to `folder`."""
    timetable = pg_database.find_recent_timetable(adjusted)
    if timetable is None:
        raise ValueError("couldn't find a timetable")

    path = shutil.copy(timetable.actual_timetable_path, folder)

    date = next_day(timetable.upload_date, 0)

    return TimetableData(timetable.id, path, date)


def get_timetable_data(
    pg_database: database.Database, id_: int, folder: pathlib.Path
) -> TimetableData:
    stmt = sqlalchemy.select(database.Timetable).where(database.Timetable.id == id_)

    with orm.Session(pg_database.engine) as session:
        result = session.execute(stmt)
        timetable = result.one().tuple()[0]
        session.expunge_all()

    if timetable is None:
        raise ValueError("couldn't find a timetable")

    path = shutil.copy(timetable.actual_timetable_path, folder)

    date = next_day(timetable.upload_date, 0)

    return TimetableData(timetable.id, path, date)


def produce_cost_metrics(
    pg_database: database.Database,
    timetable: TimetableData,
    cost_metrics_params: OTPParameters,
    zone_system_params: ZoningSystemParams,
    folder: pathlib.Path,
):
    start = datetime.datetime.now()

    with pg_database.engine.connect() as conn:
        stmt = sql.text(
            "SELECT id, zone_type_id, name, longitude, latitude\n"
            "FROM spatial_data.zones WHERE zone_type_id = :zone_system_id"
        ).bindparams(zone_system_id=zone_system_params.id)
        zone_system = pd.read_sql(stmt, conn)

    zone_system.rename(columns=_OTP_CENTROIDS_RENAME, inplace=True)
    # Save zone system CSV to assets folder
    zone_filename = f"otp_zone_centroid_id{zone_system_params.id}.csv"
    config.ASSET_DIR.mkdir(exist_ok=True)
    zone_system.to_csv(config.ASSET_DIR / zone_filename, index=False)

    # Fill in config with input file names and parameters
    otp_config = config.ProcessConfig(
        date=timetable.date,
        gtfs_files=[timetable.path.name],
        centroids=zone_filename,
        extents=zone_system_params.extents,
        osm_file=cost_metrics_params.osm_file,
        time_periods=cost_metrics_params.time_periods,
        modes=cost_metrics_params.modes,
        generalised_cost_factors=cost_metrics_params.generalised_cost_factors,
        iterinary_aggregation_method=cost_metrics_params.iterinary_aggregation_method,
        max_walk_distance=cost_metrics_params.max_walk_distance,
        number_of_threads=cost_metrics_params.number_of_threads,
        crowfly_max_distance=cost_metrics_params.crowfly_max_distance,
        write_raw_responses=False,
    )
    otp_config.save_yaml(folder / "config.yml")

    try:
        otp.run_process(folder=folder, save_parameters=False, prepare=True, force=False)
    except Exception as exc:
        run_id = pg_database.insert_run_metadata(
            database.ModelName.OTP4GB,
            start,
            otp_config.model_dump_json(),
            False,
            error=f"OTP error {exc.__class__.__name__}: {exc}",
        )
        LOG.debug("Error running OTP, logged to run metadata table with ID=%s", run_id)
        raise

    for tp in otp_config.time_periods:
        travel_datetime = datetime.datetime.combine(otp_config.date, tp.travel_time)

        for modes in otp_config.modes:
            mode = "_".join(i.name for i in modes)
            run_id = pg_database.insert_run_metadata(
                database.ModelName.OTP4GB,
                start,
                otp_config.model_dump_json(),
                True,
                output=f'{tp.name} {mode} done. Run folder: "{folder.absolute()}"',
            )

            # Find cost metrics file
            # TODO Make method for finding outputs more robust
            metrics_path = (
                folder / f"{tp.name}/{mode}_costs_{travel_datetime:%Y%m%dT%H%M}.csv"
            )
            df: pd.DataFrame = pd.read_csv(metrics_path)

            # Update column names
            df.drop(columns=_OTP_COST_METRICS_DROP_COLUMNS, inplace=True)
            df.rename(columns=_OTP_COST_METRICS_RENAME_COLUMNS, inplace=True)

            # Add run metadata ID and zone system columns
            df["run_id"] = run_id
            df["timetable_id"] = timetable.id
            df["zone_type_id"] = zone_system_params.id

            with pg_database.engine.connect() as conn:
                df.to_sql(
                    "cost_metrics",
                    conn,
                    schema="bus_data",
                    index=False,
                    if_exists="append",
                )


def main():
    params = SchedulerConfig.load_yaml(_CONFIG_FILE)
    details = ctk.ToolDetails("bus-analytics-scheduler", "0.1.0")

    with ctk.LogHelper("", details, log_file=params.output_folder / "scheduler.log"):

        # TODO Timetable IDs should be found in database using download_timetable_to_assets
        # instead of being hardcoded
        timetable_ids = [1, 3]

        pg_db = database.Database(params.database_parameters)

        for zone_system in params.zoning_system_parameters:
            folder = (
                params.output_folder
                / f"OTP {zone_system.name} - {datetime.date.today():%Y%m%d}"
            )

            for time_id in timetable_ids:
                try:
                    timetable = get_timetable_data(pg_db, time_id, folder)
                    produce_cost_metrics(
                        pg_db, timetable, params.otp_parameters, zone_system, folder
                    )

                except Exception:
                    LOG.error(
                        "Error running OTP for timetable %s", time_id, exc_info=True
                    )

            # TODO Find and delete older cached run folders


if __name__ == "__main__":
    main()
