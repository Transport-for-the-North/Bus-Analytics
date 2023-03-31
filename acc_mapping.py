import geopandas as gpd
import pandas as pd
import pandas_bokeh
import warnings
import yaml

warnings.filterwarnings("ignore")
with open("config_acc_mapping.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


shape_zone_name = config["shape_zone_name"]
zone_name = config["zone_name"]
zone_code = config["zone_code"]
LAD = config["LAD"]
shapefile = config["shapefile"]
metric = config["metric_for_plotting"]
zone_type = config["zone_type"]
mapping_periods = config["period"]
output_name = config["output_name"]
pop_data = config["pop_data_file"]
pop_zone_name = config["pop_zone_name"]
pop_zone_code = config["pop_zone_code"]
demographic = config["demographic"]
gdf = gpd.read_file(shapefile)
first_row = gpd.read_file(shapefile, rows=1)
gdf = pd.concat([first_row, gdf]).reset_index(drop=True)
gdf = gdf.set_crs(27700)
gdf = gdf.to_crs(4326)
gdf = gdf.drop(index=0).reset_index(drop=True)
gdf_1 = gdf.loc[gdf[shape_zone_name].str.rsplit(" ", 1).str[0].isin(LAD)]


pop = pd.read_csv(pop_data)
pop = pop[[pop_zone_name, pop_zone_code, demographic]]
# read shapefile
for period in mapping_periods:
    pandas_bokeh.output_file(f"{output_name}acc_metric_{metric}_{period}.html")
    acc_file = config[f"accessibility_data_{period}"]
    acc = pd.read_csv(acc_file)
    acc = acc[[zone_name, zone_code, metric]]
    acc = acc.loc[acc[zone_name].str.rsplit(" ", 1).str[0].isin(LAD)]
    acc_gdf = pd.merge(
        acc,
        gdf_1,
        left_on=zone_name,
        right_on=shape_zone_name,
        how="right",
    )
    acc_gdf = gpd.GeoDataFrame(
        data=acc_gdf.drop("geometry", axis=1), geometry=acc_gdf.geometry
    )

    acc_plot = acc_gdf.plot_bokeh(
        simplify_shapes=100,
        figsize=(900, 600),
        category=metric,
        hovertool_columns=[shape_zone_name, metric],
        colormap="Inferno",
        title=f"Accessibilty Metric {metric} for {zone_type} in the {period} period",
        legend=f"{zone_type} Zones",
        tile_provider="OSM",
        colorbar_tick_format="0a",
        alpha=0.6,
    )

    pandas_bokeh.output_file(
        f"{output_name}acc_metric_{metric}_{demographic}_{period}.html"
    )
    pop_gdf = pd.merge(
        acc_gdf, pop, left_on=zone_code, right_on=pop_zone_code, how="left"
    )
    pop_gdf[f"{metric}/{demographic}"] = pop_gdf[metric] / pop_gdf[demographic]

    pop_gdf = gpd.GeoDataFrame(
        data=pop_gdf.drop("geometry", axis=1), geometry=pop_gdf.geometry
    )
    pop_plot = pop_gdf.plot_bokeh(
        simplify_shapes=100,
        figsize=(900, 600),
        category=f"{metric}/{demographic}",
        colormap="Inferno",
        title=f"Accessibility Metric {metric}/{demographic} for {zone_type} in the {period} period",
        legend=f"{zone_type} Zones",
        tile_provider="OSM",
        colorbar_tick_format="0a",
        alpha=0.6,
    )
    pop_plot
    pd.set_option("plotting.backend", "pandas_bokeh")
