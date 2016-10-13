#from osgeo import ogr
from shapely.geometry import Polygon, shape
from fiona import collection
from fiona.crs import from_epsg
import time

#admin_file_name = 'admin3.shp'
admin_file_name = 'PHL_adm3_PSA_pn_2016June/PHL_adm3_PSA_pn_2016June.shp'
typhoon_file_name = 'typhoon_melor_edited.shp/WindSpeed.shp'
output_file_name = 'admin3_average_all.shp'


def get_average(part_list):
    if len(part_list) == 0:
        return 0.0
    total_area = 0
    total_speed = 0
    for part in part_list:
        speed = float(part['speed'][:-4])
        total_area += part['area']
        total_speed += (part['area'] * speed)
    return total_speed / total_area

t0 = time.time()

with collection(admin_file_name, 'r') as admin_file:
    output_schema = {'geometry': 'Polygon', 'properties': {u'OBJECTID': 'int', u'Mun_Code': 'str', u'average_speed': 'float'}}

    with collection(output_file_name, 'w', "ESRI Shapefile", output_schema, crs=from_epsg(4326)) as output_file:
        for admin_feat in admin_file:
            print 'admin_feat: ' + str(admin_feat['properties'][u'OBJECTID'])
            part_list = []
            admin_shp = shape(admin_feat['geometry'])
            with collection(typhoon_file_name, 'r') as typhoon_file:
                for wind_feat in typhoon_file:
                    #print '  wind_feat'
                    #print wind_feat['properties']
                    part = admin_shp.intersection(shape(wind_feat['geometry']))
                    #print part.area
                    if part.area > 0:
                        part_list.append({'area': part.area, 'speed': wind_feat['properties']['Name']})
                #print part_list
                print get_average(part_list)

            output_file.write({
                'properties': {
                    u'OBJECTID': admin_feat['properties'][u'OBJECTID'],
                    u'Mun_Code':admin_feat['properties'][u'Mun_Code'],
                    u'average_speed': get_average(part_list)
                },
                'geometry': admin_feat['geometry']
            })

t1 = time.time()

print 'total time (s): ' + str(round((t1 - t0), 3))
