import partridge as ptg
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
from shapely.geometry import LineString
import warnings
import sys

warnings.filterwarnings("ignore")
import math
import re
import yaml


def read_config(
    filepath="C:/Users/UKJMH006/Documents/TfN/Stage-2/Bus-Analytics/config_new.yml",
):
    """Reads and parses a configuration file named 'config.yml' containing user defined setting for this script.

    Raises:
        ValueError: If any of the config keys are missing or contain missing values.
        FileNotFoundError: If the 'config.yml' file in not in the working folder
        yaml.YAMLError: If there is an error while parsing the YAML content.

    Returns:
        dict: A dictionary containing the config settings

    The function reads the 'config.yml' file and ensures that it contains the following required keys:
    - 'granularity': The zone system used to identify origin and destination zones. LSOA or MSOA.
    - 'OD_zones': Filepaths to a .csv file which contains two columns. One for the origin zones and one for destination zones.
    - 'gtfs_filepath': The value specifying the full filepath to the GTFS folder.
    - 'output_file': The value specifying the fullfilepath name output file path.

    If any of these keys are missing or have empty values, a ValueError is raised. If the 'config.yml'
    file is not found or there is an error in parsing the YAML content, appropriate exceptions are raised.
    Once the configuration is successfully read and validated, it is returned as a dictionary.

    """
    try:
        with open(
            "C:/Users/UKJMH006/Documents/TfN/Stage-2/Bus-Analytics/config_new.yml"
        ) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            required_keys = [
                "granularity",
                "OD_zones",
                "file_gtfs",
                "output_file",
            ]
            for key in required_keys:
                if key not in config or config[key] is None:
                    raise ValueError(
                        f"Missing of empty value for required config key: {key}"
                    )
        return config
    except FileNotFoundError:
        print("Error: the 'config.yml' file was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error reading 'config.yml': {e}")
        sys.exit(1)


def read_zones_shapefile(shapefile):
    ######KIND POINTLESS TO DISTINGUISH BETWEEN GRANULARITIES HERE FIX IT#############
    """Reads in the UK zone shapefile of the granularity specified in the config file.


    Returns:
        GeoDataFrame: GeoDataFrame of the zone system specified by the user, ready to be filtered into areas that are of interest. in CRS(4326) (Long/Lat)
    """

    shapefile = config["shapefile"]
    gdf = gpd.read_file(shapefile)
    first_row = gpd.read_file(shapefile, rows=1)
    gdf = pd.concat([first_row, gdf]).reset_index(drop=True)
    gdf = gdf.set_crs(27700)
    gdf = gdf.to_crs(4326)
    gdf = gdf.drop(index=0).reset_index(drop=True)
    return gdf


def trip_selection(s_t_df):
    # trips that reach both zones
    trips = s_t_df.groupby("trip_id", group_keys=False)["orig/dest"].apply(
        lambda x: len(list(np.unique(x)))
    )
    trips = pd.DataFrame({"trip_id": trips.index, "trip_selection": trips.values})
    od_trips = trips[trips["trip_selection"] == 2].reset_index(drop=True)
    print(od_trips)
    # trip that go for orig to destination zone at some point in the trip
    # to be imporved to avoid a for loop in future, for the moment is fast enough
    for trip in od_trips["trip_id"]:
        # for each trip, order the stop sequences
        check = s_t_df[s_t_df["trip_id"] == trip]
        check = check.sort_values(by=["stop_sequence"])
        # if the orig/dest value increased, return 1. Circular routes will be kept
        check["increasing"] = np.where(
            check["orig/dest"] > check["orig/dest"].shift(), 1, 0
        )
        right_direction = 1 in check["increasing"].values
        # add True or False to od_trips dataframe if the trip goes from origin to destination
        od_trips.loc[od_trips.trip_id == trip, "direction"] = right_direction
    # filter trips
    final_od_trips = od_trips[od_trips["direction"] == True].reset_index(drop=True)
    # return new view with trips that only go from origin to dest
    view_1 = {"trips.txt": {"trip_id": final_od_trips["trip_id"]}}
    return view_1


def combine_tables(stop_times, stops, trips, routes, cutoffs=[0, 7, 10, 16, 19, 24]):
    """
    Combine tables containing stop times, stop information, trip information, and route information.

    Args:
    - stop_times (DataFrame): DataFrame containing stop times information.
    - stops (DataFrame): DataFrame containing stop information.
    - trips (DataFrame): DataFrame containing trip information.
    - routes (DataFrame): DataFrame containing route information.
    - cutoffs (list): List of cutoff times in hours.

    Returns:
    - stops_trips_times_routes (GeoDataFrame): GeoDataFrame containing combined information from all tables.
    """
    # Adjusting departure times exceeding 24 hours
    stop_times["departure_time"] %= 24 * 3600

    stop_times["departure_time"] /= 3600
    stop_times["arrival_time"] /= 3600

    labels = []
    for w in cutoffs:
        if float(w).is_integer():
            label = f"{w:02d}:00"
        else:
            hours = int(w)
            minutes = int((w - hours) * 60)
            label = f"{hours:02d}:{minutes:02d}"
        labels.append(label)

    labels = [f"{labels[i]}-{labels[i + 1]}" for i in range(len(labels) - 1)]

    # Put each trip into the right period
    stop_times["period"] = pd.cut(
        stop_times["departure_time"], bins=cutoffs, right=False, labels=labels
    )
    stop_times = stop_times.dropna(subset=["period"])

    # Cut the hour into bins
    hours = list(range(25))
    hours_labels = [f"{h:02d}:00" for h in range(24)]
    stop_times["hour"] = pd.cut(
        stop_times["departure_time"], bins=hours, right=False, labels=hours_labels
    )

    # Merging tables
    trips_times = pd.merge(
        stop_times,
        trips[["route_id", "trip_id", "shape_id", "trip_headsign"]],
        how="left",
        on="trip_id",
    )
    trips_times_routes = pd.merge(
        trips_times,
        routes[["route_id", "agency_id", "route_short_name", "route_type"]],
        how="left",
        on="route_id",
    )

    stops_trips_times_routes = pd.merge(
        stops[["stop_id", "stop_name", "geometry"]],
        trips_times_routes[
            [
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_id",
                "stop_sequence",
                "shape_dist_traveled",
                "period",
                "hour",
                "route_id",
                "shape_id",
                "trip_headsign",
                "agency_id",
                "route_short_name",
                "route_type",
            ]
        ],
        how="left",
        on="stop_id",
    )

    # Convert to GeoDataFrame
    stops_trips_times_routes = gpd.GeoDataFrame(
        data=stops_trips_times_routes.drop("geometry", axis=1),
        geometry=stops_trips_times_routes.geometry,
    )

    # Rename columns
    stops_trips_times_routes.rename(
        columns={"period": "stop_time_period", "hour": "stop_time_hour"}, inplace=True
    )

    return stops_trips_times_routes


def journey_times(stops_trips_times_routes, od_stops, cutoffs=[0, 7, 10, 16, 19, 24]):
    stop_times = stops_trips_times_routes.copy()
    # Adjusting departure times exceeding 24 hours
    if max(cutoffs) <= 24:
        stop_times.loc[stop_times["departure_time"] >= 24, "departure_time"] -= 24
    else:
        stop_times["departure_time"] %= 24

    # Creating labels for time periods
    def format_label(time):
        hours = int(time)
        minutes = int((time - hours) * 60)
        return f"{hours:02d}:{minutes:02d}"

    labels = [format_label(w) for w in cutoffs]

    # Creating time period labels
    period_labels = [f"{labels[i]}-{labels[i+1]}" for i in range(len(labels) - 1)]

    # Merging stop_times with origin/destination stops
    jt_cals = pd.merge(
        stop_times, od_stops[["stop_id", "orig/dest"]], how="left", on="stop_id"
    )
    jt_cals["orig/dest"].fillna(0, inplace=True)

    # Filtering origin and destination stops
    jt_orig = jt_cals[jt_cals["orig/dest"] == 1]
    jt_dest = jt_cals[jt_cals["orig/dest"] == 2]

    # Calculating start and end times for trips
    jt_start = jt_orig.groupby("trip_id")["departure_time"].min().reset_index()
    jt_end = jt_dest.groupby("trip_id")["arrival_time"].max().reset_index()

    # Merging start and end times
    jt = pd.merge(jt_start, jt_end, on="trip_id")
    jt["jt"] = jt["arrival_time"] - jt["departure_time"]

    # Assigning time periods to trips
    jt["trip_period"] = pd.cut(
        jt["departure_time"], bins=cutoffs, right=False, labels=period_labels
    ).astype(str)
    jt = jt.dropna(subset=["trip_period"])

    # Assigning hour labels to trips
    jt["trip_hour"] = pd.cut(
        jt["departure_time"],
        bins=list(range(25)),
        right=False,
        labels=[f"{i:02d}:00" for i in range(24)],
    ).astype(str)

    # Renaming columns
    jt.rename(
        columns={
            "arrival_time": "trip_arrival_time",
            "departure_time": "trip_departure_time",
        },
        inplace=True,
    )

    # Merging journey time data with original data
    stops_trips_times_routes_jt = pd.merge(
        stops_trips_times_routes,
        jt[["trip_id", "jt", "trip_period", "trip_hour"]],
        how="left",
        on="trip_id",
    )
    print(stops_trips_times_routes_jt.columns)
    print(stops_trips_times_routes_jt.dtypes)
    stops_trips_times_routes_jt.to_csv("help.csv")
    return stops_trips_times_routes_jt


def create_shapes(no_shape_df, stop_lines_1, jt_1):
    # groupby each trip, with stops in order of stop sequence. By deffinition the stop sequeunces must be in ascending order
    stop_lines = no_shape_df.groupby("trip_id", group_keys=False).apply(
        lambda x: x.sort_values("stop_sequence")
    )
    # for each trip, create a list of stop_ids and geometry in order of stop sequence
    stop_lines_stops = stop_lines.groupby("trip_id", group_keys=False)["stop_id"].apply(
        lambda x: x.tolist() if x.size > 1 else x.tolist()
    )
    stop_lines_shapes = stop_lines.groupby("trip_id", group_keys=False)[
        "geometry"
    ].apply(lambda x: LineString(x.tolist()) if x.size > 1 else x.tolist())

    stop_lines_stops = pd.DataFrame(
        {"trip_id": stop_lines_stops.index, "stop_list": stop_lines_stops.values}
    )
    stop_lines_shapes = pd.DataFrame(
        {"trip_id": stop_lines_shapes.index, "geometry": stop_lines_shapes.values}
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
    stop_lines_shapes = pd.merge(
        stop_lines_shapes, stop_lines_stops[["trip_id", "shape_id"]], on="trip_id"
    )
    # add new shapes to shapes.txt list
    total_stop_lines = pd.concat(
        [stop_lines_1, stop_lines_shapes[["shape_id", "geometry"]]]
    ).drop_duplicates()

    # merge the JT df with the df with the new shape IDs
    jt_1 = pd.merge(
        jt_1, stop_lines_shapes[["trip_id", "shape_id"]], on="trip_id", how="left"
    )
    # there are two shape_id columns after the merge. Create new one with shape_IDs taken from both columns and remove original two
    jt_1["shape_id"] = jt_1["shape_id_x"].fillna(jt_1["shape_id_y"])
    jt_1.drop(["shape_id_x", "shape_id_y"], axis=1, inplace=True)
    return total_stop_lines, jt_1


def stop_frequency(combined_table):
    trips_per_period = combined_table.pivot_table(
        "trip_id", index=["stop_id", "trip_period", "route_type"], aggfunc="count"
    ).reset_index()

    trips_per_period.rename(columns={"trip_id": "ntrips_period"}, inplace=True)
    start_time = trips_per_period["trip_period"].apply(lambda x: int(x.split(":")[0]))
    end_time = trips_per_period["trip_period"].apply(
        lambda x: int(re.search("-(.*?):", x).group(1))
    )

    trips_per_period["mean_headway(period)"] = (
        (end_time - start_time) * 60 / trips_per_period["ntrips_period"]
    ).astype(int)
    stop_frequencies = pd.merge(
        trips_per_period,
        combined_table.loc[:, ["geometry", "stop_name", "stop_id"]].drop_duplicates(
            subset=["stop_id"]
        ),
        on="stop_id",
        how="left",
    ).reset_index()

    stop_frequencies = gpd.GeoDataFrame(
        data=stop_frequencies.drop("geometry", axis=1),
        geometry=stop_frequencies.geometry,
    )
    return stop_frequencies


###############################################################################
config = read_config(sys.argv[1])

# set up wards or LSOAs being ploted

granularity = config["granularity"]
zone_name = config["zone_name_column"]
input_zones = pd.read_csv(config["OD_zones"])


feed_path = config["file_gtfs"]


# set up dictionary for filtering GTFS feed
view = {"trips.txt": {}, "stops.txt": {}}

# filter by date
service_ids_by_date = ptg.read_service_ids_by_date(feed_path)
date_1 = datetime.date(2023, 9, 20)
service_ids = service_ids_by_date[date_1]
view["trips.txt"]["service_id"] = service_ids

# reread feed with just on the sepcified date
feed = ptg.load_geo_feed(feed_path, view)
stops = feed.stops
stops = stops.set_crs(4326, allow_override=True)

# read shapefile and filter
shapefile = config["shapefile"]
gdf = read_zones_shapefile(shapefile)
split = lambda x: x[zone_name].rsplit(" ", 1)[0]


# find which stops are in the chosen origin and destinations zones
stops_in_zone = gpd.tools.sjoin(stops, gdf, op="within", how="left")
origin_zones = input_zones["origin"]
dest_zones = input_zones["dest"]
origin_zones = origin_zones.dropna()
dest_zones = dest_zones.dropna()
if origin_zones.empty:
    origin_zones = stops_in_zone[zone_name]
    origin_zones_zones = origin_zones[~origin_zones.isin(dest_zones)]
if dest_zones.empty:
    dest_zones = stops_in_zone[zone_name]
    dest_zones = dest_zones[~dest_zones.isin(origin_zones)]
origin_stops = pd.DataFrame()
dest_stops = pd.DataFrame()
# Process origin zones
# Determine the length of the DataFrame based on the longer Series
input_length = max(len(origin_zones), len(dest_zones))
origin_zones = origin_zones.reindex(range(input_length)).fillna("")
dest_zones = dest_zones.reindex(range(input_length)).fillna("")
# Create the DataFrame
input_zones = pd.DataFrame(
    {
        "origin": origin_zones,
        "dest": dest_zones,
    }
)
print(input_zones)
for ward in input_zones["origin"]:
    origin_stops = origin_stops.append(
        stops[stops_in_zone[zone_name] == ward], ignore_index=True
    )


# Add Zone_stop_id and orig/dest columns to origin stops
origin_stops["Zone_stop_id"] = range(10000, 10000 + len(origin_stops))
origin_stops["orig/dest"] = 1

# Process destination zones
for ward in input_zones["dest"]:
    dest_stops = dest_stops.append(
        stops[stops_in_zone[zone_name] == ward], ignore_index=True
    )

# Add Zone_stop_id and orig/dest columns to destination stops
dest_stops["Zone_stop_id"] = range(20000, 20000 + len(dest_stops))
dest_stops["orig/dest"] = 2
# Combine origin and destination stops
od_stops = pd.concat([origin_stops, dest_stops]).reset_index(drop=True)

# Update view with stops in origin and destination zones
view["stops.txt"]["stop_id"] = od_stops["stop_id"]
# reread feed with only stops in OD zones for given date
feed = ptg.load_geo_feed(feed_path, view=view)
# load stop times and join with OD information
stop_times = feed.stop_times
stops = feed.stops
# filters the stop_times table to remove stops that are not in the OD zones
stops_and_times = stop_times.merge(
    od_stops[["stop_id", "stop_name", "Zone_stop_id", "orig/dest", "geometry"]],
    how="inner",
    on="stop_id",
)

view = trip_selection(stops_and_times)
# read in final feed for each period and link routes trips agency table
feed_1 = ptg.load_geo_feed(feed_path, view=view)

stops_OD_trips = feed_1.stops.set_crs(4326, allow_override=True)
stops_OD_trips = pd.merge(
    stops_OD_trips, od_stops.loc[:, ["stop_id", "orig/dest"]], how="left", on="stop_id"
)
stops_OD_trips["orig/dest"].fillna(0, inplace=True)
stop_times = feed_1.stop_times

gdf_2 = gdf[gdf[zone_name].isin(input_zones["origin"])]
origin_area = gdf_2.dissolve()
origin_area[zone_name] = "Origin"
gdf_3 = gdf[gdf[zone_name].isin(input_zones["dest"])]
dest_area = gdf_3.dissolve()
dest_area[zone_name] = "Destination"
od_area = pd.concat([origin_area, dest_area]).reset_index(drop=True)
od_area = od_area.rename(
    columns={
        zone_name: "Zone",
    }
)


stop_lines_1 = feed_1.shapes.set_crs(4326, allow_override=True)

view = trip_selection(stops_and_times)
# read in final feed for each period and link routes trips agency table
feed_1 = ptg.load_geo_feed(feed_path, view=view)


trips = feed_1.trips
routes = feed_1.routes
stops_trips_times_routes = combine_tables(stop_times, stops_OD_trips, trips, routes)


jt = journey_times(stops_trips_times_routes, od_stops, cutoffs=[0, 7, 10, 16, 19, 24])

groupby = jt.groupby(["trip_period", "route_type"], group_keys=True).agg(
    {"trip_id": "nunique", "jt": ["mean", "min", "max"]}
)
# setting up hover infomation for plots
groupby.columns = ["_".join((col[0], str(col[1]))) for col in groupby.columns]
groupby = groupby.reset_index()


no_shape = jt[jt["shape_id"].isna()]
if not no_shape.empty:
    shapes_test, jt_1 = create_shapes(no_shape, stop_lines_1, jt)
    # find the number of buses with the same route, and shape within a period
    trips_per_period = jt_1.groupby(
        ["shape_id", "route_short_name", "trip_period"], group_keys=True
    ).agg({"trip_id": "nunique", "jt": "sum"})
    trips_per_period = trips_per_period.add_suffix("_Count").reset_index()
    groupby_2 = (
        jt_1.groupby(["shape_id", "trip_hour", "trip_period", "route_short_name"])
        .agg({"trip_id": "nunique", "jt": "sum"})
        .reset_index()
    )
    max_trips = groupby_2.pivot_table(
        "trip_id", index=["shape_id", "trip_period", "route_short_name"], aggfunc="max"
    ).reset_index()
    max_trips.rename(columns={"trip_id": "ntrips(peak hour)"}, inplace=True)
    line_frequencies = pd.merge(trips_per_period, max_trips, how="left")
    # add geometries
    line_frequencies = pd.merge(line_frequencies, shapes_test, on="shape_id")
    line_frequencies["shape_id"] = line_frequencies["shape_id"].astype(str)
    # covert into geodataframe
    line_frequencies = gpd.GeoDataFrame(
        data=line_frequencies.drop("geometry", axis=1),
        geometry=line_frequencies.geometry,
    ).set_crs(4326, allow_override=True)
else:
    trips_per_period = jt.groupby(
        ["shape_id", "route_short_name", "trip_period"], group_keys=True
    ).agg({"trip_id": "nunique", "jt": "sum"})
    trips_per_period = trips_per_period.add_suffix("_Count").reset_index()
    groupby_2 = (
        jt.groupby(["shape_id", "trip_hour", "trip_period", "route_short_name"])
        .agg({"trip_id": "nunique", "jt": "sum"})
        .reset_index()
    )
    max_trips = groupby_2.pivot_table(
        "trip_id", index=["shape_id", "trip_period", "route_short_name"], aggfunc="max"
    ).reset_index()
    max_trips.rename(columns={"trip_id": "ntrips(peak hour)"}, inplace=True)
    line_frequencies = pd.merge(trips_per_period, max_trips, how="left")
    # add geometries
    line_frequencies = pd.merge(line_frequencies, stop_lines_1, on="shape_id")
    line_frequencies["shape_id"] = line_frequencies["shape_id"].astype(str)
    # covert into geodataframe
    line_frequencies = gpd.GeoDataFrame(
        data=line_frequencies.drop("geometry", axis=1),
        geometry=line_frequencies.geometry,
    ).set_crs(4326, allow_override=True)


output_file = config["output_file"]
line_frequencies.rename(
    columns={"trip_id_Count": "ntrips(period)", "jt_Count": "Aggregated_jt_Mins"},
    inplace=True,
)
line_frequencies["Average_jt(period)"] = (
    line_frequencies["Aggregated_jt_Mins"] / line_frequencies["ntrips(period)"]
)
output_columns = [
    "trip_period",
    "route_short_name",
    "shape_id",
    "ntrips(period)",
    "Aggregated_jt_Mins",
    "Average_jt(period)",
    "ntrips(peak hour)",
    "geometry",
]

output_name = config["output_name"]
shapes = line_frequencies.reset_index(drop=True).set_crs(4326, allow_override=True)
shapes = shapes.sort_values("route_short_name").reset_index(drop=True)
shapes = shapes.loc[
    :,
    [
        "route_short_name",
        "shape_id",
        "trip_period",
        "ntrips(period)",
        "ntrips(peak hour)",
        "Aggregated_jt_Mins",
        "Average_jt(period)",
        "geometry",
    ],
]
shapes.to_file(output_name + "line_frequncies.shp", driver="ESRI Shapefile")

frequency_1 = stop_frequency(jt)
frequency_1 = frequency_1.reset_index()


frequency_1.to_file(output_name + "stops_frequncies.shp", driver="ESRI Shapefile")
