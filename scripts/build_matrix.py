import os
import time
import datetime
import rasterio
import pyproj
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from fiona.crs import from_epsg
from ftplib import FTP

current_path = os.path.dirname(os.path.abspath(__file__))

#Specify location of datasets
workspace = os.path.join(current_path, "dataset")

#Specify event specific data
typhoon_name = "Hagupit"
date_start = "05-12-2014" # For precipitation data, DD-MM-YYYY
date_end = "09-12-2014" # For precipitation data, DD-MM-YYYY

#Specify dataset file names
admin_file_name = "PHL_adm3_PSA_selection_epsg32651.shp"
windspeed_file_name = "windspeed_hagupit_epsg32651.shp"
track_file_name = "track_hagupit.shp"

#Specify output file names
output_shp_name = typhoon_name + "_matrix.shp"
output_csv_name = typhoon_name + "_matrix.csv"

#Specify PPM login
ppm_username = "bleumink@gmail.com"

#Output will be in this CRS, datasets are reprojected if necessary
force_epsg = 32651	#UTM Zone 51N

#Specify P Coded column in administrative boundaries file
p_code = "Mun_Code"

def timestamp(t1, t0):
	t_delta = t1 - t0
	print("done (%fs)" % t_delta)
	return t1

def get_latlon(coords_x, coords_y, epsg):
	projection = pyproj.Proj(from_epsg(force_epsg))
	lon, lat = projection(coords_x, coords_y, inverse=True)
	return lat, lon

def date_range(date_start, date_end):
	start_d, start_m, start_y = date_start.split('-')
	end_d, end_m, end_y = date_end.split('-')

	date_start = datetime.date(int(start_y), int(start_m), int(start_d))
	date_end = datetime.date(int(end_y), int(end_m), int(end_d))
	
	date_list = [str(date_start + datetime.timedelta(days=x)) for x in range((date_end - date_start).days + 1)]

	return date_list

def reproject_file(gdf, file_name, epsg):
	t0 = time.time()

	print("Reprojecting %s to EPSG %i..." % (file_name, epsg), end="", flush=True)
	gdf = gdf.to_crs(epsg=force_epsg)

	t1 = timestamp(time.time(), t0)

	return gdf

def download_gpm(date_start, date_end, download_path, ppm_username):
	base_url = "arthurhou.pps.eosdis.nasa.gov" 
	data_dir = "/pub/gpmdata/" # data_dir/yyyy/mm/dd/gis

	date_list = date_range(date_start, date_end)
	file_list = []

	os.makedirs(download_path, exist_ok=True)
	
	print("Connecting to %s..." % base_url, end="", flush=True)	
	with FTP(base_url) as ftp:
		ftp.login(ppm_username, ppm_username)
		print("OK!")
	
		for date in date_list:
			t0 = time.time()
			print("Retrieving GPM data for %s..." % date, end="", flush=True)

			d, m, y = reversed(date.split('-'))
			day_path = os.path.join(download_path, y+m+d)			
			os.makedirs(day_path, exist_ok=True)

			ftp.cwd(os.path.join(data_dir, y, m, d, 'gis'))
			for entry in ftp.mlsd():
				file_name = entry[0]
				if file_name.endswith(('tif', 'tfw')) and entry[0][3:6] == 'DAY':
					file_path = os.path.join(day_path, file_name)
					if file_name.endswith('tif'): 
						file_list.append(file_path)
						if os.path.isfile(file_path): print("found locally...", end="", flush=True)
					if not os.path.isfile(file_path):
						with open(file_path, 'wb') as write_file: 
							ftp.retrbinary('RETR ' + file_name, write_file.write)

			t1 = timestamp(time.time(), t0)
			
	return file_list

def download_srtm(bounding_box, download_path):
	base_url = "srtm.csi.cgiar.org"
	data_dir = "SRTM_V41/SRTM_Data_GeoTiff"

	tile_y0 = int((bounding_box[0][0] + 60) // 5)
	tile_y1 = int((bounding_box[0][1] + 60) // 5)
	tile_x0 = int((bounding_box[1][0] + 180) // 5)
	tile_x1 = int((bounding_box[1][1] + 180) // 5)

	zip_list = []
	os.makedirs(download_path, exist_ok=True)

	print("Connecting to %s..." % base_url, end="", flush=True)	
	with FTP(base_url) as ftp:
		ftp.login()
		print("OK!")

		ftp.cwd(data_dir)

		for x in range(tile_x0, tile_x1 + 1):
			for y in range(tile_y0, tile_y1 + 1):
				t0 = time.time()
				file_name = "srtm_%i_%i.zip" % (x, y)
				print("Retrieving %s..." % file_name, end="", flush=True)

				file_path = os.path.join(download_path, file_name)
				zip_list.append(file_path)
				if not os.path.isfile(file_path):
					with open(file_path, 'wb') as write_file: 
						ftp.retrbinary('RETR ' + file_name, write_file.write)

				t1 = timestamp(time.time(), t0)

	return

def average_windspeed(admin_geometry, windspeed_geometry, windspeed_value):
	wind_shapes = ((geometry, value) for geometry, value in zip(windspeed_geometry, windspeed_value))
	part_list = []
	for wind_shape in wind_shapes:
		part = admin_geometry.intersection(wind_shape[0])
		if part.area > 0:
			part_list.append((part.area, float(wind_shape[1][:-4])))

		if len(part_list) > 0:
			total_area = sum(i for i, j in part_list)
			total_speed = sum(i * j for i, j in part_list)
			avg_speed = total_speed / total_area
		else: avg_speed = 0

	return avg_speed

def extract_coast(admin_geometry):
	dissolve_poly = admin_geometry.unary_union
	coast_line = dissolve_poly.boundary
	cp_ratios = []

	for admin_shape in admin_geometry:
		perimeter = admin_shape.boundary
		intersect = perimeter.intersection(coast_line)
		ratio = intersect.length / perimeter.length
		cp_ratios.append(ratio)

	return cp_ratios

def cumulative_rainfall(admin_geometry, date_start, date_end, download_path, ppm_username):
	file_list = download_gpm(date_start, date_end, download_path, ppm_username)
	
	t0 = time.time()	
	print("Reading GPM data...", end="", flush=True)
	raster_list = []
	for input_raster in file_list:
		with rasterio.open(input_raster) as src:
			array = src.read(1)
			transform = src.affine
		array[array == 9999] = 0
		raster_list.append(array)

	t1 = timestamp(time.time(), t0)

	print("Calculating cumulative rainfall...", end="", flush=True)
	sum_raster = np.add.reduce(raster_list)
	sum_raster = sum_raster / 10 * 24

	admin_transform = admin_geometry.to_crs(epsg=4326)
	sum_rainfall = zonal_stats(admin_transform, sum_raster, stats='mean', nodata=-999, all_touched=True, affine=transform)
	sum_rainfall = [i['mean'] for i in sum_rainfall]

	t1 = timestamp(time.time(), t0)

	return sum_rainfall

def geography(admin_geometry, download_path):
	total_bounds = admin_geometry.total_bounds
	bounding_box = get_latlon((total_bounds[0], total_bounds[2]), (total_bounds[1], total_bounds[3]), force_epsg)
	elevation_raster = download_srtm(bounding_box, download_path)

	return

t0 = time.time()

gpm_path = os.path.join(workspace, "GPM")
srtm_path = os.path.join(workspace, "SRTM")

admin_file = os.path.join(workspace, admin_file_name)
windspeed_file = os.path.join(workspace, windspeed_file_name)
track_file = os.path.join(workspace, track_file_name)
output_shp_file = os.path.join(workspace, output_shp_name)
output_csv_file = os.path.join(workspace, output_csv_name)

# Loading shapefiles
print("Importing shapefiles...", end="", flush=True)
admin_gdf = gpd.GeoDataFrame.from_file(admin_file)
windspeed_gdf = gpd.GeoDataFrame.from_file(windspeed_file)
track_gdf = gpd.GeoDataFrame.from_file(track_file)

t1 = timestamp(time.time(), t0)

# Check CRS of each layer and reproject if necessary
if int(admin_gdf.crs['init'].split(':')[1]) != force_epsg: admin_gdf = reproject_file(admin_gdf, admin_file_name, force_epsg)
if int(windspeed_gdf.crs['init'].split(':')[1]) != force_epsg: windspeed_gdf = reproject_file(windspeed_gdf, windspeed_file_name, force_epsg)
if int(track_gdf.crs['init'].split(':')[1]) != force_epsg: track_gdf = reproject_file(track_gdf, track_file_name, force_epsg)

t1 = time.time()

output_columns = [
'P_Code',
'avg_speed',
'dist_track',
'cp_ratio',
'rainfall',
'avg_elev',
'avg_slope',
'area_km2',
'geometry']

output_gdf = gpd.GeoDataFrame(columns=output_columns, crs=from_epsg(force_epsg))

print("Assigning P codes...", end="", flush=True)
output_gdf['P_Code'] = admin_gdf[p_code]
t1 = timestamp(time.time(), t1)

print("Calculating average windspeeds...", end="", flush=True)
output_gdf['avg_speed'] = admin_gdf.geometry.apply(average_windspeed, args=(windspeed_gdf.geometry, windspeed_gdf['Name']))
t1 = timestamp(time.time(), t1)

print("Calculating centroid distances...", end="", flush=True)
output_gdf['dist_track'] = admin_gdf.centroid.geometry.apply(lambda g: track_gdf.distance(g).min()) / 10 ** 3
t1 = timestamp(time.time(), t1)

# print("Calculating coastline intersections...", end="", flush=True)
# output_gdf['cp_ratio'] = extract_coast(admin_gdf.geometry)
# t1 = timestamp(time.time(), t1)

print("Calculating areas...", end="", flush=True)
output_gdf['area_km2'] = admin_gdf.area / 10 ** 6
output_gdf.geometry = admin_gdf.geometry
t1 = timestamp(time.time(), t1)

output_gdf['rainfall'] = cumulative_rainfall(admin_gdf.geometry, date_start, date_end, gpm_path, ppm_username)

output_gdf['avg_elev'], 
output_gdf['avg_slope'] = geography(admin_gdf.geometry, srtm_path)

t1 = time.time()

# Save output as shapefile and csv
print("Exporting output to %s..." % output_shp_name, end="", flush=True)
output_gdf.to_file(output_shp_file)
t1 = timestamp(time.time(), t1)

if output_csv_name:
	print("Exporting output to %s..." % output_csv_name, end="", flush=True)
	output_df = output_gdf.drop('geometry', axis=1)
	output_df.to_csv(output_csv_file)
	t1 = timestamp(time.time(), t1)

t_total = time.time()
print('Completed in %fs' % (t_total - t0))
