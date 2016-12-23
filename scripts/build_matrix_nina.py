import os
import zipfile
import rasterio
import numpy as np
import datetime as dt
import geopandas as gpd
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, RESAMPLING
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
admin_file_name = "PHL_adm3_PSA_pn_2016June.shp"
windspeed_file_name = "windspeed_hagupit.shp"
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
	t_delta = (t1 - t0).total_seconds()
	print("done (%fs)" % t_delta)
	return t1

# def to_latlon(coords_x, coords_y, epsg):
# 	projection = pyproj.Proj(from_epsg(force_epsg))
# 	lon, lat = projection(coords_x, coords_y, inverse=True)
# 	return lat, lon

def date_range(date_start, date_end):
	start_d, start_m, start_y = date_start.split('-')
	end_d, end_m, end_y = date_end.split('-')

	date_start = dt.date(int(start_y), int(start_m), int(start_d))
	date_end = dt.date(int(end_y), int(end_m), int(end_d))
	
	date_list = [str(date_start + dt.timedelta(days=x)) for x in range((date_end - date_start).days + 1)]

	return date_list

def unzip(zip_file, destination):
	os.makedirs(destination, exist_ok=True)

	with zipfile.ZipFile(zip_file) as zf:
		zf.extractall(destination)

	return

def reproject_file(gdf, file_name, epsg):
	t0 = dt.datetime.now()

	print("Reprojecting %s to EPSG %i..." % (file_name, epsg), end="", flush=True)
	gdf = gdf.to_crs(epsg=force_epsg)

	t1 = timestamp(dt.datetime.now(), t0)

	return gdf

def reproject_raster(src_array, src_transform, src_epsg, dst_epsg, src_nodata=-32768, dst_nodata=-32768):
	src_height, src_width = src_array.shape
	dst_affine, dst_width, dst_height = calculate_default_transform(
		from_epsg(src_epsg), 
		from_epsg(dst_epsg), 
		src_width, 
		src_height, 
		*array_bounds(src_height, src_width, src_transform))

	dst_array = np.zeros((dst_width, dst_height))
	dst_array.fill(dst_nodata)

	reproject(
		src_array,
		dst_array,
		src_transform=src_transform,
		src_crs=from_epsg(src_epsg),
		dst_transform=dst_affine,
		dst_crs=from_epsg(dst_epsg),
		src_nodata=src_nodata,
		dst_nodata=dst_nodata,
		resampling=RESAMPLING.nearest)
	
	return dst_array, dst_affine

def slope(array, transform):
	height, width = array.shape
	bounds = array_bounds(height, width, transform)

	cellsize_x = (bounds[2] - bounds[0]) / width
	cellsize_y = (bounds[3] - bounds[1]) / height

	z = np.zeros((height + 2, width + 2))
	z[1:-1,1:-1] = array
	dx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2*cellsize_x)
	dy = (z[2:,1:-1] - z[:-2, 1:-1]) / (2*cellsize_y)

	slope_deg = np.arctan(np.sqrt(dx*dx + dy*dy)) * (180 / np.pi)

	return slope_deg

def ruggedness(array, transform):
	from scipy import signal

	kernel_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	kernel_y = kernel_x.transpose()
	dx = signal.convolve(array,kernel_x,mode='valid')
	dy = signal.convolve(array,kernel_y,mode='valid')

	tr_index = np.sqrt(dx**2 + dy**2)

	return tr_index

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
			t1 = dt.datetime.now()
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

			t1 = timestamp(dt.datetime.now(), t1)
			
	return file_list

def download_srtm(bounding_box, download_path):
	base_url = "srtm.csi.cgiar.org"
	data_dir = "SRTM_V41/SRTM_Data_GeoTiff"

	tile_x0 = int((bounding_box[0] + 180) // 5) + 1
	tile_x1 = int((bounding_box[2] + 180) // 5) + 1
	tile_y0 = int((60 - bounding_box[3]) // 5) + 1
	tile_y1 = int((60 - bounding_box[1]) // 5) + 1

	tif_list = []
	zip_list = []
	ignore_list =[]

	t1 = dt.datetime.now()

	print("Checking local cache for SRTM tiles...", end="", flush=True)

	ignore_file = os.path.join(download_path, "ignore_tiles.txt")
	if os.path.isfile(ignore_file):
		with open(ignore_file, 'r') as file:
			for line in file.readlines():
				ignore_list.append(line.strip())

	for x_int in range(tile_x0, tile_x1 + 1):
		for y_int in range(tile_y0, tile_y1 + 1):
			if x_int > 9: x = str(x_int)
			else: x = "0" + str(x_int)
			if y_int > 9: y = str(y_int)
			else: y = "0" + str(y_int)

			tile_folder = os.path.join(download_path, "%s_%s" % (x, y))
			tile_path = os.path.join(tile_folder, "srtm_%s_%s.tif" % (x, y))
			zip_path = os.path.join(download_path, "srtm_%s_%s.zip" % (x, y))

			if os.path.isfile(tile_path): tif_list.append(tile_path)
			else: 
				if "%s_%s" % (x, y) not in ignore_list: 
					zip_list.append((tile_folder, tile_path, zip_path, x, y))

	total_tiles = len(tif_list) + len(zip_list)
	print("found %i of %i tiles..." % (len(tif_list), total_tiles), end="", flush=True)
	t1 = timestamp(dt.datetime.now(), t1)

	if zip_list:
		print("Connecting to %s..." % base_url, end="", flush=True)
		with FTP(base_url) as ftp:
			ftp.login()
			print("OK!")
			ftp.cwd(data_dir)
	
			os.makedirs(download_path, exist_ok=True)

			for tile_folder, tile_path, zip_path, x, y in list(zip_list):
				t1 = dt.datetime.now()

				zip_name = os.path.basename(zip_path)
				print("Retrieving %s..." % zip_name, end="", flush=True)

				if not os.path.isfile(zip_path):
					with open(zip_path, 'wb') as write_file: 
						try: 
							ftp.retrbinary('RETR ' + zip_name, write_file.write)
						except: 
							print("skipped...", end="", flush=True)
							os.remove(zip_path)
							zip_list.remove((tile_folder, tile_path, zip_path, x, y))
							ignore_list.append("%s_%s" % (x,y))
				
				else: print("found locally...", end="", flush=True)

				t1 = timestamp(dt.datetime.now(), t1)

		if ignore_list:
			with open(ignore_file, 'w') as file:
				for tile in ignore_list:
					file.write(tile + '\n')

	if zip_list:
		print("Unzipping downloaded tiles...", end="", flush=True)
		for tile_folder, tile_path, zip_path, x, y in zip_list: 
			unzip(zip_path, tile_folder)
			tif_list.append(tile_path)

		t1 = timestamp(dt.datetime.now(), t1)

	return tif_list

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
	coast_length = []
	cp_ratio = []

	for admin_shape in admin_geometry:
		perimeter = admin_shape.boundary
		intersect = perimeter.intersection(coast_line)
		ratio = intersect.length / perimeter.length
		coast_length.append(intersect.length)
		cp_ratio.append(ratio)

	return coast_length, cp_ratio

def cumulative_rainfall(admin_geometry, date_start, date_end, download_path, ppm_username):
	file_list = download_gpm(date_start, date_end, download_path, ppm_username)
	
	t1 = dt.datetime.now()	
	print("Reading GPM data...", end="", flush=True)
	raster_list = []
	for input_raster in file_list:
		with rasterio.open(input_raster) as src:
			array = src.read(1)
			transform = src.affine
		array[array == 9999] = 0
		raster_list.append(array)

	t1 = timestamp(dt.datetime.now(), t1)

	print("Calculating cumulative rainfall...", end="", flush=True)
	sum_raster = np.add.reduce(raster_list)
	sum_raster = sum_raster / 10 * 24

	sum_rainfall = zonal_stats(admin_geometry, sum_raster, stats='mean', nodata=-999, all_touched=True, affine=transform)
	sum_rainfall = [i['mean'] for i in sum_rainfall]

	t1 = timestamp(dt.datetime.now(), t1)

	return sum_rainfall

def srtm_features(admin_geometry, bounding_box, download_path):
	file_list = download_srtm(bounding_box, download_path)	

	t1 = dt.datetime.now()	
	print("Reading SRTM data...", end="", flush=True)

	raster_list = []

	for input_raster in file_list: 
		raster_list.append(rasterio.open(input_raster))

	if len(raster_list) > 1: 
		srtm_dem, srtm_transform = merge(raster_list, nodata=-32768)
		srtm_dem = srtm_dem[0]
	else: 
		srtm_dem = raster_list[0].read(1)
		srtm_transform = raster_list[0].affine

	for input_raster in raster_list:
		input_raster.close()
	del(raster_list)
	t1 = timestamp(dt.datetime.now(), t1)

	print("Reprojecting DEM to EPSG %i..." % force_epsg, end="", flush=True)
	srtm_utm, transform_utm = reproject_raster(srtm_dem, srtm_transform, 4326, force_epsg, -32768, 0)
	t1 = timestamp(dt.datetime.now(), t1)

	print("Calculating mean elevation...", end="", flush=True)
	avg_elevation = zonal_stats(admin_geometry, srtm_utm, stats='mean', nodata=-32768, all_touched=True, affine=transform_utm)
	avg_elevation = [i['mean'] for i in avg_elevation]
	t1 = timestamp(dt.datetime.now(), t1)

	print("Calculating mean slope...", end="", flush=True)
	avg_slope = zonal_stats(admin_geometry, slope(srtm_utm, transform_utm), stats='mean', nodata=0, all_touched=True, affine=transform_utm)
	avg_slope = [i['mean'] for i in avg_slope]
	t1 = timestamp(dt.datetime.now(), t1)

	print("Calculating mean ruggedness...", end="", flush=True)
	avg_rugged = zonal_stats(admin_geometry, ruggedness(srtm_utm, transform_utm), stats='mean', nodata=0, all_touched=True, affine=transform_utm)
	avg_rugged = [i['mean'] for i in avg_rugged]
	t1 = timestamp(dt.datetime.now(), t1)

	return avg_elevation, avg_slope, avg_rugged

t0 = dt.datetime.now()

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

t1 = timestamp(dt.datetime.now(), t0)

# Check if CRS is defined and default to WGS 84 if not
if not admin_gdf.crs: admin_gdf.crs = from_epsg(4326)
if not windspeed_gdf.crs: windspeed_gdf.crs = from_epsg(4326)
if not track_gdf.crs: track_gdf.crs = from_epsg(4326)

# Keeping an unprojected copy of admin area geometry in WGS84 to speed up raster calculations
if int(admin_gdf.crs['init'].split(':')[1]) != 4326: admin_geometry_wgs84 = reproject_file(admin_gdf.geometry, admin_file_name, 4326)
else: admin_geometry_wgs84 = admin_gdf.geometry

# Check CRS of each layer and reproject if necessary
if int(admin_gdf.crs['init'].split(':')[1]) != force_epsg: admin_gdf = reproject_file(admin_gdf, admin_file_name, force_epsg)
if int(windspeed_gdf.crs['init'].split(':')[1]) != force_epsg: windspeed_gdf = reproject_file(windspeed_gdf, windspeed_file_name, force_epsg)
if int(track_gdf.crs['init'].split(':')[1]) != force_epsg: track_gdf = reproject_file(track_gdf, track_file_name, force_epsg)

t1 = dt.datetime.now()

output_columns = [
'P_Code',
'avg_speed',
'dist_track',
'coast_len',
'cp_ratio',
'rainfall',
'avg_elev',
'avg_slope',
'avg_rugged',
'area_km2',
'geometry']

output_gdf = gpd.GeoDataFrame(columns=output_columns, crs=from_epsg(force_epsg))

# Comment out sections here if you don't need to calculate them

print("Assigning P codes...", end="", flush=True)
output_gdf['P_Code'] = admin_gdf[p_code]
t1 = timestamp(dt.datetime.now(), t1)

print("Calculating average windspeeds...", end="", flush=True)
output_gdf['avg_speed'] = admin_gdf.geometry.apply(average_windspeed, args=(windspeed_gdf.geometry, windspeed_gdf['Name']))
t1 = timestamp(dt.datetime.now(), t1)

print("Calculating centroid distances...", end="", flush=True)
output_gdf['dist_track'] = admin_gdf.centroid.geometry.apply(lambda g: track_gdf.distance(g).min()) / 10 ** 3
t1 = timestamp(dt.datetime.now(), t1)

# print("Calculating coastline intersections...", end="", flush=True)
# output_gdf['coast_len'], output_gdf['cp_ratio'] = extract_coast(admin_gdf.geometry)
# t1 = timestamp(dt.datetime.now(), t1)

# print("Calculating areas...", end="", flush=True)
# output_gdf['area_km2'] = admin_gdf.area / 10 ** 6
# t1 = timestamp(dt.datetime.now(), t1)

# Calculating cumulative rainfall
output_gdf['rainfall'] = cumulative_rainfall(admin_geometry_wgs84, date_start, date_end, gpm_path, ppm_username)

# Calculating terrain features
#output_gdf['avg_elev'], output_gdf['avg_slope'], output_gdf['avg_rugged'] = srtm_features(admin_gdf.geometry, admin_geometry_wgs84.total_bounds, srtm_path)

# Assigning geometry
output_gdf.geometry = admin_gdf.geometry

t1 = dt.datetime.now()

# Save output as shapefile and csv
print("Exporting output to %s..." % output_shp_name, end="", flush=True)
output_gdf.to_file(output_shp_file)
t1 = timestamp(dt.datetime.now(), t1)

if output_csv_name:
	print("Exporting output to %s..." % output_csv_name, end="", flush=True)
	output_df = output_gdf.drop('geometry', axis=1)
	output_df.to_csv(output_csv_file)
	t1 = timestamp(dt.datetime.now(), t1)

t_total = dt.datetime.now()
print('Completed in %fs' % (t_total - t0).total_seconds())