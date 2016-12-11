import os
import time
import fiona
from fiona.crs import from_epsg
from shapely.geometry import Polygon, shape
from reproject_shp import reproject

def check_projections(file_list):
	check_crs = []
	for file in file_list:
		with fiona.open(file) as f:
			crs = int(list(f.crs.values())[0].split(':')[1])
			if crs not in check_crs: check_crs.append(crs)
	if len(check_crs) > 1: return False
	else: return crs

def average_windspeed(admin_shp, windspeed_file):
	part_list = []
	for wind_feat in windspeed_file:
		part = admin_shp.intersection(shape(wind_feat['geometry']))
		if part.area > 0:
			part_list.append({'area': part.area, 'speed': wind_feat['properties']['Name']})

	total_area = 0.
	total_speed = 0.

	for part in part_list:
		speed = float(part['speed'][:-4])
		total_area += part['area']
		total_speed += (part['area'] * speed)
	if total_speed == 0.: avg_speed = 0
	else: avg_speed = total_speed / total_area

	return avg_speed 

def centroid_distance(admin_shp, track_file):
   	centroid = admin_shp.centroid
   	track_shape = shape(track_file[0]['geometry'])
   	return centroid.distance(track_shape) / 1000


t0 = time.time()

workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

admin_file_name = os.path.join(workspace, "PHL_adm3_PSA_selection_epsg32651.shp")
windspeed_file_name = os.path.join(workspace, "windspeed_hagupit_epsg32651.shp")
track_file_name = os.path.join(workspace, "track_hagupit_epsg32651.shp")
output_file_name = os.path.join(workspace, "output_test.shp")

infile_list = [admin_file_name, windspeed_file_name, track_file_name]

force_epsg = 32651	#UTM Zone 51N
proj4 = "+proj=utm +zone=51 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

crs_match = check_projections(infile_list)

if not crs_match or (force_epsg and crs_match != force_epsg):
	if not crs_match:
		print("Input CRS do not match, reprojecting...")
	elif force_epsg and crs_match != force_epsg:
		print("Input CRS does not match specified projection, reprojecting...")
		
	infile_list = reproject(infile_list, force_epsg, proj4)
	admin_file_name = infile_list[0]
	windspeed_file_name = infile_list[1]
	track_file_name = infile_list[2]

with fiona.open(admin_file_name, 'r') as admin_file:

	num_records = len(admin_file)
	dict_col = {u'OBJECTID': 'int',
	u'P_Code': 'str',
	u'avg_windspeed_mph': 'float',
	u'distance_km': 'float',
	u'area_km2': 'float'}

	output_schema = {'geometry': 'Polygon', 'properties': dict_col}

	with fiona.open(windspeed_file_name, 'r') as windspeed_file:
		with fiona.open(track_file_name, 'r') as track_file:
			with fiona.open(output_file_name, 'w', "ESRI Shapefile", output_schema, crs=from_epsg(force_epsg)) as output_file:
			    for admin_feat in admin_file:

			    	admin_shp = shape(admin_feat['geometry'])

			    	distance = centroid_distance(admin_shp, track_file)
			    	speed = average_windspeed(admin_shp, windspeed_file)			    	
			    	area =  admin_shp.area / 10 ** 6 

			    	output_file.write({
			    		'properties': {
			    		u'OBJECTID': admin_feat['properties'][u'OBJECTID'],
			    		u'P_Code':admin_feat['properties'][u'Mun_Code'],
			    		u'avg_speed': speed,
			    		u'dist_track': distance,
			    		u'area_km2': area,},
			    		'geometry': admin_feat['geometry']
			    		})

t1 = time.time()
print('total time (s):', round(t1 - t0, 3))