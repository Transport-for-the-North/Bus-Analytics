# -*- coding: utf-8 -*-
"""
    Dummy example running OTP process
"""

##### IMPORTS #####

import logging
import os
import pathlib
import shutil
import sys

import pandas as pd
from bodse import database
from sqlalchemy import sql

# OTP4GB dependency needs to be manually installed and setup from GitHub https://github.com/transport-for-the-north/otp4gb-py/
# Environment variable can be used to define local repo location, although default should suffice
OTP4GB_REPO_FOLDER = os.environ.get("OTP4GB_REPO_FOLDER", "../otp4gb-py")
sys.path.append(OTP4GB_REPO_FOLDER)
from otp4gb import config, otp  # pylint: disable=wrong-import-position

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


##### CLASSES & FUNCTIONS #####


def produce_cost_metrics(
    cost_metrics_params, db_params: database.DatabaseConfig, zone_system_id: int
):

    pg_db = database.Database(db_params)
    # Get adjusted / non-adjusted timetable with
    timetable = pg_db.find_recent_timetable()
    if timetable is None:
        raise ValueError("something's gone wrong")
    # Save GTFS file to assets folder
    shutil.copy(timetable.actual_timetable_path, config.ASSET_DIR)

    with pg_db.engine.connect() as conn:
        stmt = sql.text(
            "SELECT id, longitude, latitude FROM spatial_data.zones WHERE zone_type_id = :zone_system_id"
        ).bindparams(zone_system_id=zone_system_id)
        zone_system = pd.read_sql(stmt, conn)

    # Save zone system CSV to assets folder
    zone_filename = f"otp_zone_centroid_id{zone_system_id}.csv"
    config.ASSET_DIR.mkdir(exist_ok=True)
    zone_system.to_csv(config.ASSET_DIR / zone_filename, index=False)

    # Create new run folder, in a cache location on C drive
    folder = pathlib.Path()
    folder.mkdir()

    # Fill in config with input file names and parameters
    otp_config = config.ProcessConfig(
        gtfs_files=[timetable.actual_timetable_path.name],
        centroids=zone_filename,
        date=NotImplemented,
    )
    otp_config.save_yaml(folder / "config.yml")

    otp.run_process(folder=folder, save_parameters=False, prepare=True, force=False)

    # Insert run metadata for the process
    run_id = pg_db.insert_run_metadata()

    for tp in otp_config.time_periods:
        # Find cost metrics file
        # Read CSV and add time period column, maybe update column names
        # Add run metadata ID column
        metrics_path = folder / f"{tp.name}/{mode}_cost_metrics.csv"
        df: pd.DataFrame = pd.read_csv(metrics_path)
        df["time_period"] = tp.name
        df["run_metadata_id"] = run_id

        with pg_db.engine.connect() as conn:
            df.to_sql(
                "cost_metrics", conn, schema="bus_data", index=False, if_exists="append"
            )

    # Find and delete older cached run folders
