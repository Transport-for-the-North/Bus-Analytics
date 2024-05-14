# import pandas as pd
import geopandas as gpd
import partridge as ptg
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import LineString
from tqdm import tqdm

tqdm.pandas()
import yaml

# Load the YAML configuration file
with open(
    "C:/Users/UKJMH006/Documents/TfN/Stage-2/Bus-Analytics/config_new.yml", "r"
) as stream:
    try:
        config = yaml.safe_load(stream)
        print("Configuration loaded successfully!")
    except yaml.YAMLError as exc:
        print("Error loading configuration:", exc)
#######################################################
# Initialisation
#######################################################
file_stops_select = ""

study_area_name = config["study_area_name"]
file_gtfs = config["file_gtfs"]
file_polygon_select = config["file_polygon_select"]
file_full_trip_table_csv = config["file_full_trip_table_csv"]
file_select_trip_table_byday_csv = config["file_select_trip_table_byday_csv"]
file_select_trip_table_byhour_csv = config["file_select_trip_table_byhour_csv"]
file_select_trip_table_byhour_peakdir_csv = config[
    "file_select_trip_table_byhour_peakdir_csv"
]
file_select_trip_table_shape_route_stats_csv = config[
    "file_select_trip_table_shape_route_stats_csv"
]
file_select_trip_table_shape_area_stats_csv = config[
    "file_select_trip_table_shape_area_stats_csv"
]
file_select_stop_stats_csv = config["file_select_stop_stats_csv"]
go_filter_gtfs_area = config["go_filter_gtfs_area"]
go_filter_gtfs_stops = config["go_filter_gtfs_stops"]
go_filter_gtfs_stops_area = config["go_filter_gtfs_stops_area"]
go_filter_gtfs_routetype = config["go_filter_gtfs_routetype"]
go_trim_full_dates_services = config["go_trim_full_dates_services"]
go_full_trip_table_csv = config["go_full_trip_table_csv"]
go_create_stn_dep_timetable = config["go_create_stn_dep_timetable"]
accepted_modes = config["accepted_modes"]
repdayofweek_text = config["repdayofweek_text"]
select_gtfs_start_date = config["select_gtfs_start_date"]
select_gtfs_end_date = config["select_gtfs_end_date"]
select_daily_start_date = config["select_daily_start_date"]
select_daily_end_date = config["select_daily_end_date"]
select_daily_start_time = config["select_daily_start_time"]
select_daily_end_time = config["select_daily_end_time"]
select_hourly_start_date = config["select_hourly_start_date"]
select_hourly_end_date = config["select_hourly_end_date"]
select_hourly_start_time = config["select_hourly_start_time"]
select_hourly_end_time = config["select_hourly_end_time"]
select_stn_dep_date = config["select_stn_dep_date"]
select_stops_serv_freq_date = config["select_stops_serv_freq_date"]

#######################################################
# Import GTFS data
#######################################################
# Load GTFS feed
feed = ptg.load_geo_feed(file_gtfs)
view = {"trips.txt": {}, "stops.txt": {}}
# Print summary
print("******** GTFS DATA SET SUMMARY ********")
print("# stop_times:", len(feed.stop_times))
print("# stops:", len(feed.stops))
print("# routes:", len(feed.routes))
print("# calendar:", len(feed.calendar))
print("# calendar_dates:", len(feed.calendar_dates))
print("***************************************")

# Extract dataframes from feed
stop_times_df = feed.stop_times
stops_df = feed.stops
routes_df = feed.routes
calendar_df = feed.calendar
calendar_dates_df = feed.calendar_dates

#######################################################
# S03 - [OPTIONAL] Filter GTFS data set
#######################################################
go_filter_gtfs_dataset = max(go_filter_gtfs_area, go_filter_gtfs_stops)
if go_filter_gtfs_dataset > 0:
    #######################################################
    # Filter 1: Select routes that pass through area covered by the GIS polygon
    #######################################################
    if go_filter_gtfs_area == 1:
        # Load GIS polygon
        gis_polygon = gpd.read_file(file_polygon_select)
        stops_df.set_crs("EPSG:4326", inplace=True, allow_override=True)
        stops_in_zone = stops_df.clip(gis_polygon)
        stops_and_times = stop_times_df.merge(
            stops_in_zone[["stop_id", "stop_name", "geometry"]],
            how="inner",
            on="stop_id",
        )
        view["trips.txt"]["trip_id"] = stops_and_times["trip_id"]

        out_gtfs = "C:/Users/UKJMH006/Documents/TfN/Stage-2/Bus-Analytics/GTFS_out/"
        ptg.extract_feed(file_gtfs, out_gtfs, view)
        # Print summary
        feed = ptg.load_geo_feed(out_gtfs)
        stop_times_df = feed.stop_times
        stops_df = feed.stops
        routes_df = feed.routes
        calendar_df = feed.calendar
        calendar_dates_df = feed.calendar_dates
        # Print summary
        print("******** FILTER 1 (BY AREA) SUMMARY ********")
        print("# stop_times:", len(stop_times_df))
        print("# stops:", len(stops_df))
        print("# routes:", len(routes_df))
        print("# calendar:", len(calendar_df))
        print("# calendar_dates:", len(calendar_dates_df))
        print("********************************************")

    #######################################################
    # Filter 2: Select stops
    #######################################################
    print(stops_df)
    if go_filter_gtfs_stops == 1:
        if go_filter_gtfs_stops_area == 1:
            # Load GIS polygon
            gis_polygon = gpd.read_file(file_polygon_select)

            # Filter stops by GIS polygon
            stops_select_df = stops_df[
                (~stops_df["stop_lat"].isna()) & (~stops_df["stop_lon"].isna())
            ]
            stops_select_gdf = gpd.GeoDataFrame(
                stops_select_df,
                geometry=gpd.points_from_xy(
                    stops_select_df.stop_lon, stops_select_df.stop_lat
                ),
            )
            stops_select_gdf = gpd.overlay(
                stops_select_gdf, gis_polygon, how="intersection"
            )

            # Get the filtered stop_ids
            stops_filtered = list(stops_select_gdf["stop_id"])

        else:
            # Load a list of stop_ids from a CSV file
            stops_filtered = pd.read_csv(
                file_stops_select, header=None, names=["stop_id"]
            )

        # Filter GTFS feed by selected stops
        view["stops.txt"]["stop_id"] = stops_select_gdf["stop_id"]
        feed = ptg.load_feed(file_gtfs, view=view)
        stop_times_df = feed.stop_times
        stops_df = feed.stops
        routes_df = feed.routes
        calendar_df = feed.calendar
        calendar_dates_df = feed.calendar_dates
        out_gtfs = "C:/Users/UKJMH006/Documents/TfN/Stage-2/Bus-Analytics/GTFS_out/"
        # ptg.extract_feed(file_gtfs, out_gtfs, view)
        # Print summary
        print("******** FILTER 2 (BY STOPS) SUMMARY ********")
        print("# stop_times:", len(feed.stop_times))
        print("# stops:", len(feed.stops))
        print("# routes:", len(feed.routes))
        print("# calendar:", len(feed.calendar))
        print("# calendar_dates:", len(feed.calendar_dates))
        print("********************************************")

        # Continue with data processing steps...

    #######################################################
    # Filter 3: Select route types (modes)
    #######################################################
    if go_filter_gtfs_routetype == 1:
        # Define accepted modes
        accepted_modes = [3]  # Modify as needed

        # Filter GTFS feed by accepted route types
        feed = ptg.feed(
            filtered_dfs={
                "stop_times": stop_times_df,
                "stops": stops_df,
                "routes": routes_df[routes_df["route_type"].isin(accepted_modes)],
                "calendar": calendar_df,
                "calendar_dates": calendar_dates_df,
            }
        )

        # Print summary
        print("******** FILTER 3 (BY ROUTE TYPES) SUMMARY ********")
        print("# stop_times:", len(feed.stop_times))
        print("# stops:", len(feed.stops))
        print("# routes:", len(feed.routes))
        print("# calendar:", len(feed.calendar))
        print("# calendar_dates:", len(feed.calendar_dates))
        print("********************************************")


#######################################################
# S04 - Create day-of-week/holiday lookup table
#######################################################

# Create a DataFrame with unique dates from the GTFS data
service_ids_by_date = ptg.read_service_ids_by_date(file_gtfs)
service_ids_by_date_1 = pd.DataFrame(
    service_ids_by_date.items(), columns=["Date", "Service IDs"]
)
service_ids_by_date_1 = service_ids_by_date_1.explode("Service IDs")

# Reset the index to have consecutive integers as index
service_ids_by_date_1.reset_index(drop=True, inplace=True)

# Rename the columns if necessary
service_ids_by_date_1.columns = ["date", "service_id"]
service_ids_by_date_1["date"] = pd.to_datetime(service_ids_by_date_1["date"])
service_ids_by_date_1["dayofweek"] = service_ids_by_date_1["date"].dt.strftime("%A")

print(service_ids_by_date_1)
bank_holidays = pd.read_json(path_or_buf="https://www.gov.uk/bank-holidays.json")


def get_england_and_wales(data_frame):
    return pd.json_normalize(
        data_frame.to_dict(), record_path=[["england-and-wales", "events"]]
    ).astype(
        {
            "title": "string",
            "date": "datetime64[ns]",
            "notes": "string",
            "bunting": "bool",
        }
    )


bh = get_england_and_wales(bank_holidays)
print(bh)
bh["holiday"] = 1
bh = bh[["date", "holiday"]]
dayofweek_holiday_def = service_ids_by_date_1.merge(bh, on="date", how="left")


n_unique_service = dayofweek_holiday_def["service_id"].nunique()


#######################################################
# S06 - Create lookup table linking trip_id with first/last stop_ids
#######################################################
print(feed.stop_times.info())
# trip_start_time = feed.stop_times.groupby("trip_id").agg("min")
print(feed.stop_times)
# First, group by 'trip_id' and summarize 'first_stop_seq'
trip_start_time = (
    feed.stop_times.groupby("trip_id").agg({"stop_sequence": "min"}).reset_index()
)
trip_start_time.rename(columns={"stop_sequence": "first_stop_seq"}, inplace=True)

# Left join 'trip_start_time' with 'gtfs' on the specified columns
trip_start_time = pd.merge(
    trip_start_time,
    feed.stop_times,
    left_on=["trip_id", "first_stop_seq"],
    right_on=["trip_id", "stop_sequence"],
    how="left",
)

# Rename columns as specified
trip_start_time.rename(
    columns={
        "stop_id": "first_stop_id",
        "arrival_time": "first_stop_arv_time",
        "departure_time": "first_stop_dep_time",
    },
    inplace=True,
)
print(trip_start_time)

# First, group by 'trip_id' and summarize 'last_stop_seq'
trip_end_time = (
    feed.stop_times.groupby("trip_id").agg({"stop_sequence": "max"}).reset_index()
)
trip_end_time.rename(columns={"stop_sequence": "last_stop_seq"}, inplace=True)

# Left join 'trip_end_time' with 'gtfs' on the specified columns
trip_end_time = pd.merge(
    trip_end_time,
    feed.stop_times,
    left_on=["trip_id", "last_stop_seq"],
    right_on=["trip_id", "stop_sequence"],
    how="left",
)

# Rename columns as specified
trip_end_time.rename(
    columns={
        "stop_id": "last_stop_id",
        "arrival_time": "last_stop_arv_time",
        "departure_time": "last_stop_dep_time",
    },
    inplace=True,
)
print(trip_end_time)

if "shape_dist_traveled" in trip_end_time.columns:
    trip_end_time = trip_end_time.rename(
        columns={"shape_dist_traveled": "shp_dist_traveled"}
    )
else:
    trip_end_time["shp_dist_traveled"] = pd.NA
#######################################################
# S07 - Calculate journey distance according to the shape geometry
#######################################################
##############################################################################################################################################################
##############################################################################################################################################################
#############################################################################################################################################################
# calculate journey distances based on shape geometry.


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude
    in decimal degrees.

    Parameters:
    - lat1, lon1: Latitude and longitude of point 1
    - lat2, lon2: Latitude and longitude of point 2

    Returns:
    - Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of Earth in kilometers (mean value)
    radius = 6371.0

    # Calculate the distance
    distance = radius * c

    return distance


def calculate_rolling_distance(df):
    # Create a new column 'shp_dist_traveled' and initialize it with zeros
    df["shp_dist_traveled"] = 0.0
    # Calculate Haversine distance between consecutive stops
    df["lat_shift"] = (
        df["geometry"].shift(1).apply(lambda x: x.coords[0][0] if x else None)
    )
    df["lon_shift"] = (
        df["geometry"].shift(1).apply(lambda x: x.coords[0][1] if x else None)
    )

    df["shp_dist_traveled"] = df.progress_apply(
        lambda row: haversine(
            row["lat_shift"],
            row["lon_shift"],
            row["geometry"].coords[0][0],
            row["geometry"].coords[0][1],
        ),
        axis=1,
    )
    mask = df.groupby("trip_id").cumcount() == 0
    df.loc[mask, "shp_dist_traveled"] = 0

    df.to_csv("C:/Users/UKJMH006/Documents/misc_tasks/distances.csv")
    df["shp_dist_traveled"] = df.groupby("trip_id")["shp_dist_traveled"].cumsum()
    # Group by 'shape_id' and sum the distances
    print(df.head())
    # Drop the temporary columns used for shift
    df = df.drop(["lat_shift", "lon_shift"], axis=1)
    df.to_csv("C:/Users/UKJMH006/Documents/misc_tasks/test.csv")
    return df


# GTFS data set might not come with shapes (incl. the early version of BODS)
def create_shapes(feed):
    stops = feed.stops
    trips = feed.trips
    shapes = feed.shapes.set_crs(4326, allow_override=True)
    stop_times = feed.stop_times
    routes = feed.routes
    shapes = shapes.merge(trips[["trip_id", "shape_id"]], on="shape_id")

    times_stops = stops[["stop_id", "geometry"]].merge(
        stop_times[["trip_id", "stop_id", "stop_sequence"]].drop_duplicates(),
        how="inner",
        on="stop_id",
    )
    trips_times_stops = times_stops.merge(
        trips[["route_id", "trip_id", "trip_headsign", "shape_id"]],
        how="inner",
        on="trip_id",
    )
    routes_trips_times_stops = trips_times_stops.merge(
        routes[["route_id", "agency_id", "route_short_name", "route_type"]],
        how="inner",
        on="route_id",
    )
    routes_trips_times_stops = routes_trips_times_stops[
        routes_trips_times_stops["shape_id"].isna()
    ]
    # groupby each trip, with stops in order of stop sequence. By deffinition the stop sequeunces must be in ascending order
    stop_lines = routes_trips_times_stops.groupby("trip_id", group_keys=False).apply(
        lambda x: x.sort_values("stop_sequence")
    )

    # for each trip, create a list of stop_ids and geometry in order of stop sequence
    stop_lines_stops = stop_lines.groupby("trip_id", group_keys=False)["stop_id"].apply(
        lambda x: x.tolist() if x.size > 1 else x.tolist()
    )
    stop_lines = calculate_rolling_distance(stop_lines)

    stop_lines_shapes = stop_lines.groupby("trip_id", group_keys=False).agg(
        {
            "geometry": lambda x: LineString(x.tolist()) if x.size > 1 else x.tolist(),
            "shp_dist_traveled": "max",
        }
    )

    stop_lines_stops = pd.DataFrame(
        {"trip_id": stop_lines_stops.index, "stop_list": stop_lines_stops.values}
    )

    # combine lists of stop lists into one string to act as a shape ID. Trips with stops in same sequence will share shape ID
    stop_lines_stops["stop_list"] = [
        ",".join(map(str, l)) for l in stop_lines_stops["stop_list"]
    ]

    stop_lines_shapes = gpd.GeoDataFrame(stop_lines_shapes, geometry="geometry")
    stop_lines_shapes = stop_lines_shapes.set_crs(4326)
    stop_lines_stops["shape_id"] = stop_lines_stops.groupby("stop_list").ngroup()
    stop_lines_stops["shape_id"] = "manual_shape" + stop_lines_stops["shape_id"].astype(
        str
    )

    # merges df of new shape IDs with df with linestings of coordinates so that each trip ID will now have a shape ID and geometry values
    stop_lines_shapes = pd.merge(stop_lines_shapes, stop_lines_stops, on="trip_id")
    # print(stop_lines_shapes.columns)
    stop_lines_shapes.to_csv(
        "C:/Users/UKJMH006/Documents/misc_tasks/stop_lines_shapes.csv"
    )
    first_row = shapes.iloc[:1]
    shapes_1 = pd.concat([first_row, shapes]).reset_index(drop=True)
    # shapes_1 = shapes_1.set_crs(4326, allow_override=True)
    shapes_reproj = shapes_1.to_crs(27700)
    shapes_reproj = shapes_reproj.drop(index=0).reset_index(drop=True)
    # shapes["shp_dist_traveled"] = shapes_reproj["geometry"].length
    shapes["shp_dist_traveled"] = (shapes_reproj["geometry"].length) / 1000

    # add new shapes to shapes.txt list
    total_stop_lines = pd.concat(
        [
            shapes,
            stop_lines_shapes[["trip_id", "shape_id", "shp_dist_traveled", "geometry"]],
        ]
    ).drop_duplicates()
    # merge the JT df with the df with the new shape IDs

    return total_stop_lines


new_shapes = create_shapes(feed)
print(new_shapes)
new_shapes.to_csv("C:/Users/UKJMH006/Documents/misc_tasks/newshapes.csv")
if go_trim_full_dates_services == 1:
    full_dates_services = dayofweek_holiday_def[
        (dayofweek_holiday_def["date"] >= select_gtfs_start_date)
        & (dayofweek_holiday_def["date"] <= select_gtfs_end_date)
    ]
#######################################################
# S08 - Create full trip table
#######################################################
# Left join 'trips' with 'full_dates_services' on 'service_id'
full_trip_table = pd.merge(feed.trips, full_dates_services, on="service_id", how="left")

# Left join 'full_trip_table' with 'trip_start_time' on 'trip_id'
full_trip_table = pd.merge(full_trip_table, trip_start_time, on="trip_id", how="left")

# Left join 'full_trip_table' with 'trip_end_time' on 'trip_id'
full_trip_table = pd.merge(full_trip_table, trip_end_time, on="trip_id", how="left")

# Left join 'full_trip_table' with 'routes' on 'route_id'
full_trip_table = pd.merge(full_trip_table, feed.routes, on="route_id", how="left")

# Left join 'full_trip_table' with 'agency' on 'agency_id'
full_trip_table = pd.merge(full_trip_table, feed.agency, on="agency_id", how="left")

# Check if "agency_noc" is in the columns; if not, add it with 'NA'
if "agency_noc" not in full_trip_table.columns:
    full_trip_table["agency_noc"] = "NA"

# Check if "trip_headsign" is in the columns; if not, add it with 'NA'
if "trip_headsign" not in full_trip_table.columns:
    full_trip_table["trip_headsign"] = "NA"

# Check if "trip_direction_name" is in the columns; if not, add it with 'NA'
if "trip_direction_name" not in full_trip_table.columns:
    full_trip_table["trip_direction_name"] = "NA"


# Select specific columns in 'full_trip_table'
full_trip_table = full_trip_table[
    [
        "agency_id",
        "agency_name",
        "agency_noc",
        "route_id",
        "route_short_name",
        "route_long_name",
        "route_type",
        "service_id",
        "trip_id",
        "trip_headsign",
        "trip_direction_name",
        "date",
        "dayofweek",
        "holiday",
        "first_stop_seq",
        "first_stop_id",
        "first_stop_dep_time",
        "first_stop_arv_time",
        "last_stop_seq",
        "last_stop_id",
        "last_stop_dep_time",
        "last_stop_arv_time",
        "shp_dist_traveled",
        "shape_id",
    ]
]

# Add shape distance to the 'full_trip_table', if available (when "shape_id" is available in "trips" table)
if "shape_id" in full_trip_table.columns:
    full_trip_table = pd.merge(
        full_trip_table,
        new_shapes[["trip_id", "shape_id", "shp_dist_traveled"]],
        on="trip_id",
        how="left",
    )
# full_trip_table["shp_dist_traveled"] =
else:
    full_trip_table["shape_id"] = np.nan
    full_trip_table["shp_dist_traveled"] = np.nan


full_trip_table["journey_time"] = (
    full_trip_table["last_stop_arv_time"] - full_trip_table["first_stop_dep_time"]
)
# Compute time difference and save as hh:mm:ss format
full_trip_table["shape_id"] = full_trip_table["shape_id_x"].where(
    full_trip_table["shape_id_x"].notnull(), full_trip_table["shape_id_y"]
)
full_trip_table["shp_dist_traveled"] = full_trip_table["shp_dist_traveled_x"].where(
    full_trip_table["shp_dist_traveled_x"].notnull(),
    full_trip_table["shp_dist_traveled_y"],
)
full_trip_table = full_trip_table.drop(
    ["shp_dist_traveled_x", "shp_dist_traveled_y", "shape_id_x", "shape_id_y"], axis=1
)
full_trip_table["first_stop_dep_time_hour"] = pd.to_datetime(
    full_trip_table["first_stop_dep_time"], unit="s"
).dt.hour
full_trip_table["first_stop_dep_time"] = pd.to_datetime(
    pd.to_datetime(full_trip_table["first_stop_dep_time"], unit="s").dt.strftime(
        "%H:%M:%S"
    ),
    format="%H:%M:%S",
).dt.time


full_trip_table["last_stop_arv_time"] = pd.to_datetime(
    pd.to_datetime(full_trip_table["last_stop_arv_time"], unit="s").dt.strftime(
        "%H:%M:%S"
    ),
    format="%H:%M:%S",
).dt.time


print(type(full_trip_table["first_stop_dep_time"]))


# Create a DataFrame to hold unique route types
all_modes_incl = pd.DataFrame({"route_type": full_trip_table["route_type"].unique()})

# Merge the unique route types with their names
# all_modes_incl = pd.merge(all_modes_incl, route_type_names, on="route_type", how="left")

# Sort the route types by name
# all_modes_incl = all_modes_incl.sort_values("route_type_name")

# Create a text string that lists all available modes included in the full trip table for display
# all_modes_incl_text = ", ".join(all_modes_incl["route_type_name"])

# Export 'full_trip_table' as a CSV (if needed)
if go_full_trip_table_csv == 1:
    full_trip_table.to_csv(file_full_trip_table_csv, index=False)
import time

# Record the start time
start_time = time.time()

file_select_trip_table_byday = (
    full_trip_table.groupby(["date", "agency_name", "agency_noc", "route_short_name"])
    .agg({"journey_time": "sum", "trip_id": "count"})
    .reset_index()
)
file_select_trip_table_byday.rename(
    columns={"journey_time": "Agg_JT(hours)", "trip_id": "Daily Trips"}, inplace=True
)

file_select_trip_table_byday["Agg_JT(hours)"] = file_select_trip_table_byday[
    "Agg_JT(hours)"
].div(3600)
print(file_select_trip_table_byday)
file_select_trip_table_byday.to_csv(file_select_trip_table_byday_csv, index=False)
print(full_trip_table["date"].min(), full_trip_table["date"].max())
select_trip_table = full_trip_table[
    (full_trip_table["date"] >= select_hourly_start_date)
    & (full_trip_table["date"] <= select_hourly_end_date)
    & (
        (full_trip_table["first_stop_dep_time"])
        >= datetime.strptime(select_hourly_start_time, "%H:%M:%S").time()
    )
    & (
        (full_trip_table["first_stop_dep_time"])
        <= datetime.strptime(select_hourly_end_time, "%H:%M:%S").time()
    )
]

# Choose a representative weekday and select trips for Weekday, Saturday, and Sunday only, and exclude holidays
repdayofweek_text = "Wednesday"  # Change this to your desired representative weekday
select_trip_table = select_trip_table[
    (
        (select_trip_table["dayofweek"] == "Saturday")
        | (select_trip_table["dayofweek"] == "Sunday")
        | (select_trip_table["dayofweek"] == repdayofweek_text)
    )
    & pd.isna(select_trip_table["holiday"])
]


# Create a label for the representative week of day
select_trip_table["repdayofweek"] = select_trip_table["dayofweek"]
select_trip_table["repdayofweek"] = select_trip_table["repdayofweek"].replace(
    repdayofweek_text, "Weekday"
)

# Reformat route short name to add leading space for sorting
max_shortname_len = max(
    select_trip_table["route_short_name"].apply(lambda x: len(str(x))), default=0
)
select_trip_table["route_short_name"] = select_trip_table["route_short_name"].apply(
    lambda x: f"{x:>{max_shortname_len}}"
)

# Group select trip table by hour and save as CSV
select_trip_table_byhour = (
    select_trip_table.groupby(
        [
            "agency_name",
            "agency_noc",
            "route_short_name",
            "trip_headsign",
            "date",
            "dayofweek",
            "repdayofweek",
            "first_stop_dep_time_hour",
        ]
    )
    .size()
    .reset_index(name="n")
)
select_trip_table_byhour.to_csv(file_select_trip_table_byhour_csv, index=False)
# Further group select trip table head sign
select_trip_table_byhour_peakdir = (
    select_trip_table_byhour.groupby(
        [
            "agency_name",
            "agency_noc",
            "route_short_name",
            "date",
            "dayofweek",
            "repdayofweek",
            "first_stop_dep_time_hour",
        ]
    )
    .agg(MaxFreq_Hourly_PeakDir=("n", "max"))
    .reset_index()
)

# Save the result to a CSV file
select_trip_table_byhour_peakdir.to_csv(
    file_select_trip_table_byhour_peakdir_csv, index=False
)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Time taken: {elapsed_time:.4f} seconds")


if feed.shapes is not None:
    # Set selection criteria
    print(select_trip_table.columns)

    # Compute journey speed for routes with both distance and journey time information only
    select_trip_table = select_trip_table.dropna(
        subset=["shp_dist_traveled", "journey_time"]
    )
    select_trip_table["shape_dist_km"] = select_trip_table["shp_dist_traveled"] / 1000
    select_trip_table["journey_time_hr"] = select_trip_table["journey_time"] / 3600
    select_trip_table["route_speed_kph"] = (
        select_trip_table["shape_dist_km"] / select_trip_table["journey_time_hr"]
    )
# Recode time of day and day of week into numerical form for looping
select_trip_table["time_of_day"] = pd.cut(
    select_trip_table["first_stop_dep_time_hour"],
    bins=[-np.inf, 7, 9, 16, 18, 24, 28, np.inf],
    labels=[1, 2, 3, 4, 5, 6, 1],
    ordered=False,
)

select_trip_table["day_of_week_num"] = select_trip_table["dayofweek"].map(
    {"Wednesday": 1, "Saturday": 2, "Sunday": 3}
)

# Select essential columns only
select_trip_table_route_stats = (
    select_trip_table.groupby(
        ["agency_name", "agency_noc", "route_short_name", "trip_headsign"]
    )
    .size()
    .reset_index(name="num_RepWkday_Wkend_trips")
)

# Group by time of day
k = 1
select_trip_table_area_stats = pd.DataFrame(columns=["Stats", "values"])
help = 0
if help == 0:
    for i in range(1, 4):
        for j in range(1, 7):
            select_trip_table_route_stats_tod = select_trip_table[
                (select_trip_table["day_of_week_num"] == i)
                & (select_trip_table["time_of_day"] == j)
            ]

            if not select_trip_table_route_stats_tod.empty:
                select_trip_table_route_stats_tod = (
                    select_trip_table_route_stats_tod.groupby(
                        [
                            "agency_name",
                            "agency_noc",
                            "route_short_name",
                            "trip_headsign",
                        ]
                    )
                    .agg(
                        num_tod_trips=("trip_id", "size"),
                        aggr_dist_km=("shape_dist_km", "sum"),
                        aggr_journey_time_hr=("journey_time_hr", "sum"),
                    )
                    .reset_index()
                )

                select_trip_table_route_stats_tod["aggr_avg_speed"] = (
                    select_trip_table_route_stats_tod["aggr_dist_km"]
                    / select_trip_table_route_stats_tod["aggr_journey_time_hr"]
                )

                temp_table = select_trip_table_route_stats.merge(
                    select_trip_table_route_stats_tod,
                    on=[
                        "agency_name",
                        "agency_noc",
                        "route_short_name",
                        "trip_headsign",
                    ],
                    how="left",
                )

                temp_table["num_tod_trips"] = temp_table["num_tod_trips"].fillna("")
                temp_table["aggr_dist_km"] = temp_table["aggr_dist_km"].fillna("")
                temp_table["aggr_journey_time_hr"] = temp_table[
                    "aggr_journey_time_hr"
                ].fillna("")
                temp_table["aggr_avg_speed"] = temp_table["aggr_avg_speed"].fillna("")

                select_trip_table_route_stats[f"num_{i}_{j}_tod_trips"] = temp_table[
                    "num_tod_trips"
                ]
                select_trip_table_route_stats[f"aggr_dist_km_{i}_{j}"] = temp_table[
                    "aggr_dist_km"
                ]
                select_trip_table_route_stats[f"aggr_journey_time_hr_{i}_{j}"] = (
                    temp_table["aggr_journey_time_hr"]
                )
                select_trip_table_route_stats[f"aggr_avg_speed_{i}_{j}"] = temp_table[
                    "aggr_avg_speed"
                ]
            else:
                select_trip_table_route_stats[f"num_{i}_{j}_tod_trips"] = ""
                select_trip_table_route_stats[f"aggr_dist_km_{i}_{j}"] = ""
                select_trip_table_route_stats[f"aggr_journey_time_hr_{i}_{j}"] = ""
                select_trip_table_route_stats[f"aggr_avg_speed_{i}_{j}"] = ""

            num_tod_trips_area = (
                select_trip_table_route_stats.loc[
                    select_trip_table_route_stats[f"num_{i}_{j}_tod_trips"] != "",
                    f"num_{i}_{j}_tod_trips",
                ]
                .astype(float)
                .sum()
            )
            aggr_dist_km_area = (
                select_trip_table_route_stats.loc[
                    select_trip_table_route_stats[f"aggr_dist_km_{i}_{j}"] != "",
                    f"aggr_dist_km_{i}_{j}",
                ]
                .astype(float)
                .sum()
            )
            aggr_journey_time_hr_area = (
                select_trip_table_route_stats.loc[
                    select_trip_table_route_stats[f"aggr_journey_time_hr_{i}_{j}"]
                    != "",
                    f"aggr_journey_time_hr_{i}_{j}",
                ]
                .astype(float)
                .sum()
            )
            aggr_avg_speed_area = (
                aggr_dist_km_area / aggr_journey_time_hr_area
                if aggr_dist_km_area > 0 and aggr_journey_time_hr_area > 0
                else 0
            )

            name_dayofwk = "Wday" if i == 1 else "Sat" if i == 2 else "Sun"
            name_tod = (
                "Early"
                if j == 1
                else (
                    "AM"
                    if j == 2
                    else (
                        "BP"
                        if j == 3
                        else "EP" if j == 4 else "OP" if j == 5 else "Night"
                    )
                )
            )

            select_trip_table_route_stats.rename(
                columns={
                    f"num_{i}_{j}_tod_trips": f"{name_dayofwk}_{name_tod}_num_tod_trips",
                    f"aggr_dist_km_{i}_{j}": f"{name_dayofwk}_{name_tod}_aggr_dist_km",
                    f"aggr_journey_time_hr_{i}_{j}": f"{name_dayofwk}_{name_tod}_aggr_journey_time_hr",
                    f"aggr_avg_speed_{i}_{j}": f"{name_dayofwk}_{name_tod}_aggr_avg_speed",
                },
                inplace=True,
            )

            num_tod_trips_area_col = f"{name_dayofwk}_{name_tod}_num_tod_trips_area"
            aggr_dist_km_area_col = f"{name_dayofwk}_{name_tod}_aggr_dist_km_area"
            aggr_journey_time_hr_area_col = (
                f"{name_dayofwk}_{name_tod}_aggr_journey_time_hr_area"
            )
            aggr_avg_speed_area_col = f"{name_dayofwk}_{name_tod}_aggr_avg_speed_area"
            select_trip_table_area_stats = pd.concat(
                [
                    select_trip_table_area_stats,
                    pd.DataFrame(
                        {"Stats": num_tod_trips_area_col, "values": num_tod_trips_area},
                        index=[0],
                    ),
                    pd.DataFrame(
                        {"Stats": aggr_dist_km_area_col, "values": aggr_dist_km_area},
                        index=[0],
                    ),
                    pd.DataFrame(
                        {
                            "Stats": aggr_journey_time_hr_area_col,
                            "values": aggr_journey_time_hr_area,
                        },
                        index=[0],
                    ),
                    pd.DataFrame(
                        {
                            "Stats": aggr_avg_speed_area_col,
                            "values": aggr_avg_speed_area,
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )
# Writing to CSV
select_trip_table_route_stats.to_csv(
    "file_select_trip_table_shape_route_stats.csv", index=False
)
select_trip_table_area_stats.to_csv(
    "file_select_trip_table_shape_area_stats.csv", index=False
)
print("##########################DONE########################")
