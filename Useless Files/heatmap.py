import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

data = xr.open_dataset("data/delhi_cpcb_2022.nc")
df = data.to_dataframe()
# df = df["2022-01-01": "2022-12-31"]
df = data.to_dataframe().reset_index()
df = df[df["time"]=="2022-03-01 01:30:00"]
df = df.dropna(subset=["PM2.5"])

delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')
gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

latitudes = np.array(df['latitude'])
longitudes = np.array(df['longitude'])

g_lat = np.linspace(latitudes.min()-0.1, latitudes.max()+0.1, 30)
g_long = np.linspace(longitudes.min()-0.1, longitudes.max()+0.1, 30)

lat_grid, lon_grid = np.meshgrid(g_lat, g_long)
temp_data = gpd.GeoDataFrame(geometry = gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()))

def plot_heatmap(lat, lon, values): #lat, lon, values are numpy arrays

    delhi_shapefile = gpd.read_file('data/Delhi/Districts.shp')

    shapefile_extent = delhi_shapefile.total_bounds

    fig, ax = plt.subplots(figsize=(10, 10))

#     plt.xlim(shapefile_extent[0], shapefile_extent[2])
#     plt.ylim(shapefile_extent[1], shapefile_extent[3])

    contour = ax.contourf(lon.reshape(lon_grid.shape), lat.reshape(lon_grid.shape), values.reshape(lon_grid.shape), cmap='coolwarm', levels = 200)

    # Add the shapefile to the plot
    delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')

    gdf_data.plot(ax=ax, color='black', markersize=20, label='Air Stations')
#     temp_data.plot(ax=ax, color='black', markersize=20, label='Predicted Points')

    # Add a colorbar
    plt.colorbar(contour, label='PM2.5',shrink=0.7)

    # Customize the plot appearance
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('PM2.5 Predictions Heatmap')
    plt.legend()
    plt.show()