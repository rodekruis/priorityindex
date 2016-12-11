import os
import fiona
from subprocess import call

def insert_epsg(string, separator, epsg):
	index = string.index(separator)
	return string[:index] + '_epsg' + str(epsg) + string[index:]

def reproject(file_list, epsg, proj4):
	file_list_update = []

	for file in file_list:
		with fiona.open(file) as f:
			crs = int(list(f.crs.values())[0].split(':')[1])
			if crs != epsg:
				print(file, "EPSG:", crs, "Reprojecting to:", epsg)
				outfile = insert_epsg(file, '.', epsg)
				call(["ogr2ogr", 
					"-f", 
					"ESRI Shapefile", 
					"-overwrite", 
					outfile,
					file,
					"-t_srs",
					str(proj4)])
				file_list_update.append(outfile)
			else:
				print(file, "EPSG:", epsg, "OK!")
				file_list_update.append(file)
	
	return file_list_update
	
if __name__ == '__main__':
	file_list = []

	workspace = os.path.dirname(os.path.abspath(__file__))
	formats = (".shp")

	epsg = 32651	#UTM Zone 51N
	proj4 = "+proj=utm +zone=51 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"

	for file in os.listdir(workspace):
	    if file.endswith(formats):
	        file_list.append(os.path.join(workspace, file))

	reproject(file_list, epsg, proj4)