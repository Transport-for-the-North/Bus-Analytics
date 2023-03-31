import partridge as ptg
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
from shapely.geometry import LineString
import warnings

warnings.filterwarnings("ignore")
import math
import re
import yaml
import pandas_bokeh
import bokeh.palettes
from bokeh.palettes import all_palettes


pd.set_option("plotting.backend", "pandas_bokeh")
with open("config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def read_zones_shapefile(granularity):
    if granularity == "LSOA":
        # Get Census Boundaries as GeoPandas
        shapefile = config["shapefile"]
        gdf = gpd.read_file(shapefile)
        first_row = gpd.read_file(shapefile, rows=1)
        gdf = pd.concat([first_row, gdf]).reset_index(drop=True)
        gdf = gdf.set_crs(27700)
        gdf = gdf.to_crs(4326)
        gdf = gdf.drop(index=0).reset_index(drop=True)
    else:
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

    hours = list(range(25))
    hours_labels = [str(hours[i]) + ":00" for i in range(len(hours) - 1)]

    if max(cutoffs) <= 24:
        stop_times_ok = stop_times.loc[stop_times.departure_time < 24 * 3600]
        stop_times_fix = stop_times.loc[stop_times.departure_time >= 24 * 3600]
        stop_times_fix["departure_time"] = [
            d - 24 * 3600 for d in stop_times_fix.departure_time
        ]

        stop_times = stop_times_ok.append(stop_times_fix)
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if len(str(w)) == 1:
                    l = "0" + str(w) + ":00"
                else:
                    l = str(w) + ":00"
            else:
                n = math.modf(w)
                if int(n[1]) == 1:
                    l = "0" + str(int(n[1])) + ":" + str(int(n[0] * 60))
                else:
                    l = str(int(n[1])) + ":" + str(int(n[0] * 60))
            labels = labels + [l]
    else:
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if w > 24:
                    w1 = w - 24
                    l = str(w1) + ":00"
                else:
                    if len(str(w)) == 1:
                        l = "0" + str(w) + ":00"
                    else:
                        l = str(w) + ":00"
                labels = labels + [l]
            else:
                if w > 24:
                    w1 = w - 24
                    n = math.modf(w1)
                    l = str(int(n[1])) + ":" + str(int(n[0] * 60))
                else:
                    n = math.modf(w)
                    if int(n[1]) == 1:
                        l = "0" + str(int(n[1])) + ":" + str(int(n[0] * 60))
                    else:
                        l = str(int(n[1])) + ":" + str(int(n[0] * 60))
                labels = labels + [l]

    labels = [labels[i] + "-" + labels[i + 1] for i in range(0, len(labels) - 1)]

    stop_times["departure_time"] = stop_times["departure_time"] / 3600
    stop_times["arrival_time"] = stop_times["arrival_time"] / 3600
    # Put each trips in the right period
    stop_times["period"] = pd.cut(
        stop_times["departure_time"], bins=cutoffs, right=False, labels=labels
    )
    stop_times = stop_times.loc[~stop_times.period.isnull()]
    stop_times["period"] = stop_times["period"].astype(str)
    stop_times["hour"] = pd.cut(
        stop_times["departure_time"], bins=hours, right=False, labels=hours_labels
    )
    stop_times["hour"] = stop_times["hour"].astype(str)

    trips_times = pd.merge(
        stop_times,
        trips.loc[:, ["route_id", "trip_id", "shape_id", "trip_headsign"]],
        how="left",
        on="trip_id",
    )
    trips_times_routes = pd.merge(
        trips_times,
        routes.loc[:, ["route_id", "agency_id", "route_short_name", "route_type"]],
        how="left",
        on="route_id",
    )

    stops_trips_times_routes = pd.merge(
        stops.loc[:, ["stop_id", "stop_name", "geometry"]],
        trips_times_routes.loc[
            :,
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
            ],
        ],
        how="left",
        on=["stop_id"],
    )

    stops_trips_times_routes = gpd.GeoDataFrame(
        data=stops_trips_times_routes.drop("geometry", axis=1),
        geometry=stops_trips_times_routes.geometry,
    )

    stops_trips_times_routes.rename(
        columns={"period": "stop_time_period", "hour": "stop_time_hour"}, inplace=True
    )
    # stop_frequencies.sort_values(by='mean_headway(period)', ascending=False, inplace=True)
    return stops_trips_times_routes


def journey_times(stops_trips_times_routes, od_stops, cutoffs=[0, 7, 10, 16, 19, 24]):
    stop_times = stops_trips_times_routes
    hours = list(range(25))
    hours_labels = [str(hours[i]) + ":00" for i in range(len(hours) - 1)]
    if max(cutoffs) <= 24:
        stop_times_ok = stop_times.loc[stop_times.departure_time < 24]
        stop_times_fix = stop_times.loc[stop_times.departure_time >= 24]
        stop_times_fix["departure_time"] = [
            d - 24 for d in stop_times_fix.departure_time
        ]

        stop_times = stop_times_ok.append(stop_times_fix)
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if len(str(w)) == 1:
                    l = "0" + str(w) + ":00"
                else:
                    l = str(w) + ":00"
            else:
                n = math.modf(w)
                if int(n[1]) == 1:
                    l = "0" + str(int(n[1])) + ":" + str(int(n[0] * 60))
                else:
                    l = str(int(n[1])) + ":" + str(int(n[0] * 60))
            labels = labels + [l]
    else:
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if w > 24:
                    w1 = w - 24
                    l = str(w1) + ":00"
                else:
                    if len(str(w)) == 1:
                        l = "0" + str(w) + ":00"
                    else:
                        l = str(w) + ":00"
                labels = labels + [l]
            else:
                if w > 24:
                    w1 = w - 24
                    n = math.modf(w1)
                    l = str(int(n[1])) + ":" + str(int(n[0] * 60))
                else:
                    n = math.modf(w)
                    if int(n[1]) == 1:
                        l = "0" + str(int(n[1])) + ":" + str(int(n[0] * 60))
                    else:
                        l = str(int(n[1])) + ":" + str(int(n[0] * 60))
                labels = labels + [l]

    labels = [labels[i] + "-" + labels[i + 1] for i in range(0, len(labels) - 1)]
    jt_cals = pd.merge(
        stop_times, od_stops.loc[:, ["stop_id", "orig/dest"]], how="left", on="stop_id"
    )
    jt_cals["orig/dest"].fillna(0, inplace=True)
    jt_orig = jt_cals[jt_cals["orig/dest"] == 1]
    jt_dest = jt_cals[jt_cals["orig/dest"] == 2]
    jt_start = jt_orig.pivot_table("departure_time", index=["trip_id"], aggfunc="min")
    jt_end = jt_dest.pivot_table("arrival_time", index=["trip_id"], aggfunc="max")
    jt = pd.merge(jt_start, jt_end, on=["trip_id"])
    jt["jt"] = jt["arrival_time"] - jt["departure_time"]
    # trip is put into trip_period depending on departure time
    jt["trip_period"] = pd.cut(
        jt["departure_time"], bins=cutoffs, right=False, labels=labels
    )
    jt = jt.loc[~jt.trip_period.isnull()]
    jt["trip_period"] = jt["trip_period"].astype(str)
    jt["trip_hour"] = pd.cut(
        jt["departure_time"], bins=hours, right=False, labels=hours_labels
    )
    jt["trip_hour"] = jt["trip_hour"].astype(str)
    jt.rename(
        columns={
            "arrival_time": "trip_arrival_time",
            "departure_time": "trip_departure_time",
        },
        inplace=True,
    )
    jt = jt.reset_index()
    stops_trips_times_routes_jt = pd.merge(
        stops_trips_times_routes,
        jt.loc[:, ["trip_id", "jt", "trip_period", "trip_hour"]],
        how="left",
        on="trip_id",
    )

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
LAD = config["LAD"]
# set up wards or LSOAs being ploted

granularity = config["granularity"]
if granularity == "LSOA":
    zone_name = "LSOA11NM"
    zone_code = "LSOA11CD"
    input_zones = pd.read_csv(config["OD_zones"])
elif granularity == "MSOA":
    zone_name = "MSOA21NM"
    zone_code = "MSOA21CD"
    input_zones = pd.read_csv(config["OD_zones"])
else:
    print("input correct files")

# gv.output(dpi=120, fig='svg')

feed_path = config["gtfs_filepath"]

# use sheffield zones for development
LAD = config["LAD"]
# set up dictionary for filtering GTFS feed
view = {"trips.txt": {}, "stops.txt": {}}

# filter by date
service_ids_by_date = ptg.read_service_ids_by_date(feed_path)
date_1 = datetime.date(2023, 3, 20)
service_ids = service_ids_by_date[date_1]
view["trips.txt"]["service_id"] = service_ids

# reread feed with just on the sepcified date
feed = ptg.load_geo_feed(feed_path, view)
stops = feed.stops
stops = stops.set_crs(4326, allow_override=True)

# read shapefile and filter to Sheffield LAD
gdf = read_zones_shapefile(granularity)
gdf_1 = gdf.loc[gdf[zone_name].str.rsplit(" ", 1).str[0].isin(LAD)]

# find which stops are in the chosen origin and destinations zones
stops_in_zone = gpd.tools.sjoin(stops, gdf_1, op="within", how="left")
origin_zones = input_zones["origin"]
dest_zones = input_zones["dest"]


# create dataframe with stops in origin and dest zones
count = 0
for ward in input_zones["origin"]:
    if count == 0:
        origin_stops = stops[stops_in_zone[zone_name] == ward]
    else:
        origin_stops = pd.concat(
            [origin_stops, stops[stops_in_zone[zone_name] == ward]]
        )
    count += 1
    origin_stops = origin_stops.reset_index(drop=True)
    origin_stops["Zone_stop_id"] = range(10000, 10000 + len(origin_stops))
    origin_stops["orig/dest"] = 1
count = 0
for ward in input_zones["dest"]:
    if count == 0:
        dest_stops = stops[stops_in_zone[zone_name] == ward]
    else:
        dest_stops = pd.concat([dest_stops, stops[stops_in_zone[zone_name] == ward]])
    count += 1
    dest_stops = dest_stops.reset_index(drop=True)
    dest_stops["Zone_stop_id"] = range(20000, 20000 + len(dest_stops))
    dest_stops["orig/dest"] = 2
od_stops = pd.concat([origin_stops, dest_stops]).reset_index(drop=True)

# update view with only stops in origin and dest zones
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

gdf_2 = gdf_1[gdf_1[zone_name].isin(input_zones["origin"])]
origin_area = gdf_2.dissolve()
origin_area[zone_name] = "Origin"
gdf_3 = gdf_1[gdf_1[zone_name].isin(input_zones["dest"])]
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
print(view)
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
]
line_frequencies_output = line_frequencies[output_columns]
line_frequencies_output.sort_values(
    by=["trip_period", "route_short_name"], inplace=True
)
line_frequencies_output.to_csv(output_file, index=False)


frequency_1 = stop_frequency(jt)
frequency_1 = frequency_1.reset_index()

mapping_periods = config["period"]
period_times = {"AM": "07:00-10:00", "IP": "10:00-16:00", "PM": "16:00-19:00"}
for period in mapping_periods:
    output_name = config["output_name"]
    pandas_bokeh.output_file(f"{output_name}bus_frequencies_{period}.html")
    shapes = (
        line_frequencies[line_frequencies["trip_period"] == period_times[period]]
        .reset_index(drop=True)
        .set_crs(4326, allow_override=True)
    )
    shapes = shapes.sort_values("ntrips(period)").reset_index(drop=True)
    frequency = frequency_1[frequency_1["trip_period"] == period_times[period]]
    figure_1 = shapes.plot_bokeh(
        figsize=(1200, 675),
        show_figure=False,  # <== pass figure here!
        category="ntrips(period)",
        hovertool_columns=["route_short_name", "ntrips(period)"],
        colormap=bokeh.palettes.viridis(shapes["ntrips(period)"].max() + 1),
        colormap_range=(0, shapes["ntrips(period)"].max() + 1),
        tile_provider="OSM",
        colormap_uselog=False,
        line_width=6,
        legend="Bus Routes",
        title=f"Number of buses in {period} period",
        colorbar_tick_format="0",
    )
    figure = od_area.plot_bokeh(
        figure=figure_1,
        show_figure=False,
        simplify_shapes=100,
        legend="OD Zones",
        tile_provider="OSM",
        show_colorbar=False,
        alpha=0.5,
    )
    frequency.plot_bokeh(
        simplify_shapes=10000,
        tile_provider="OSM",
        figure=figure,
        # xlim=[-170, -80],
        # ylim=[10, 70],
        # category="ntrips_period",
        title=f"Number of buses in {period} period",
        legend="Bus Stops",
        size=7,
        hovertool_columns=["stop_name", "ntrips_period"],
        show_colorbar=False,
    )
