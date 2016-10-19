#from osgeo import ogr
from shapely.geometry import Polygon, shape
from fiona import collection
from fiona.crs import from_epsg
import time
import numpy as np
import pandas as pd


#admin_file_name = 'admin3.shp'
admin_file_name = 'PHL_adm3_PSA_pn_2016June_32651proj.shp'
#admin_file_name = '../municipalities_shapefiles.shp'
typhoon_file_name = 'windspeed.shp'
track_file_name = 'track.shp'
output_file_name = 'admin3_average_all.shp'
base_path = './'
typhoon_list = ['Rammasun' , 'Melor' , 'Hagupit', 'Haiyan' ]
typhoon_list = ['Haiyan']

for typhoon_name in typhoon_list : 
    def get_average(part_list):
        total_area = 0.
        total_speed = 0.
        for part in part_list:
            speed = float(part['speed'][:-4])
            total_area += part['area']
            total_speed += (part['area'] * speed)
        if total_speed == 0.: avg_speed = 0
        else: avg_speed = total_speed / total_area

        return avg_speed 

    t0 = time.time()

    with collection(base_path+admin_file_name, 'r') as admin_file:

        print len(admin_file)
        vec_area = np.zeros(len(admin_file))
        vec_avg_speed = np.zeros(len(admin_file))
        vec_distance_typhoon = np.zeros(len(admin_file))
        vec_pcode = []
        dict_col = {u'OBJECTID': 'int', u'Mun_Code': 'str', u'average_speed_mph': 'float', u'distance_typhoon_km': 'float', u'area_km2': 'float'}
        output_schema = {'geometry': 'Polygon', 'properties': dict_col}

        with collection(base_path+typhoon_name+'/'+output_file_name, 'w', "ESRI Shapefile", output_schema, crs=from_epsg(4326)) as output_file:
            #print admin_file
            index = 0 
            for admin_feat in admin_file:

                #if index == 10 : break
                #print admin_feat
                print 'admin_feat: ' + str(admin_feat['properties'][u'OBJECTID'])
                part_list = []
                admin_shp = shape(admin_feat['geometry'])

                with collection(base_path+typhoon_name+'/'+track_file_name, 'r') as track_file:
                    centroid=admin_shp.centroid
                    print 'centroid '  ,  centroid
                    for track_feat in  track_file :
                        track_shape = shape(track_feat['geometry'])
                        distance = centroid.distance(track_shape)/1000.
                        print 'distance ' , distance

                with collection(base_path+typhoon_name+'/'+typhoon_file_name, 'r') as typhoon_file:
                    for wind_feat in typhoon_file:
                        #print '  wind_feat'
                        #print wind_feat['properties']
                        part = admin_shp.intersection(shape(wind_feat['geometry']))
                        #print part.area
                        if part.area > 0:
                            part_list.append({'area': part.area, 'speed': wind_feat['properties']['Name']})
                    #print part_list
                speed  = get_average(part_list)
                area =  admin_shp.area/ 10**6
                output_file.write({
                    'properties': {
                        u'OBJECTID': admin_feat['properties'][u'OBJECTID'],
                        u'Mun_Code':admin_feat['properties'][u'Mun_Code'],
                        u'average_speed_mph': speed,
                        u'distance_typhoon_km': distance,
                        u'area_km2': area,
                    },
                    'geometry': admin_feat['geometry']
                })


                vec_area[index] = area
                vec_avg_speed[index] = speed
                vec_distance_typhoon[index] = distance
                vec_pcode.append(admin_feat['properties'][u'Mun_Code'])


                index +=1

    t1 = time.time()

    print 'total time (s): ' + str(round((t1 - t0), 3))

    df = pd.DataFrame(columns = dict_col)
    df['Mun_Code'] = vec_pcode
    df['average_speed_mph'] = vec_avg_speed
    df['distance_typhoon_km'] = vec_distance_typhoon
    df['area_km2'] = vec_area
    df.to_csv(base_path+typhoon_name+'/data_windspeed_and_distance.csv')
