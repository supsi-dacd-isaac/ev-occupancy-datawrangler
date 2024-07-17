import os
import geopandas as gpd
import requests
import pandas as pd
import json
import wget
from shapely.geometry import Point
from tqdm import tqdm
import numpy as np
from os.path import join, exists
from itertools import product

BASE_PATH = 'data'


def query_layer(polygon, layer_name='ch.bfs.gebaeude_wohnungs_register', sr_format=4326):
    geometry = polygon.to_crs('EPSG:{}'.format(sr_format)).iloc[0]['geometry']

    # Initialize an empty list to hold the coordinates
    coordinates_list = []

    # Access the exterior coordinates of the polygon and append to the list
    coords = list(geometry.exterior.coords)
    coordinates_list.append(coords)

    # Convert the coordinates list to the format expected by the API
    geometry_param = json.dumps({"rings": coordinates_list}).replace(" ", "")

    # Ensure the geometry parameter is URL-encoded to avoid any issues in the GET request
    from urllib.parse import quote
    geometry_param_encoded = quote(geometry_param)
    minx, miny, maxx, maxy = geometry.bounds
    offset = 0
    results = []
    while True:
        #print('\rlayer{}, offset:{}'.format(layer_name, offset), end='', flush=True)
        query_url = (f"https://api3.geo.admin.ch/rest/services/api/MapServer/identify?geometry={geometry_param_encoded}"
                     f"&geometryType=esriGeometryPolygon&imageDisplay=500,600,96&mapExtent={minx},{miny},{maxx},{maxy}"
                     f"&tolerance=5&layers=all:{layer_name}&sr={sr_format}&offset={offset}&limit=50")
        response = requests.get(query_url)
        data = response.json()
        results.extend(data['results'])

        if len(data['results']) < 50:
            break
        else:
            offset += 50

    attributes_list = [item['attributes'] for item in results]
    # Creating a DataFrame from the 'attributes' list
    results_pd = pd.DataFrame(attributes_list)
    return results_pd


def get_traffic(polygon, static_traffic_df=None):
    if static_traffic_df is None:
        df = query_layer(polygon, layer_name='ch.are.belastung-personenverkehr-strasse')
        if df.empty or df is None:
            return None
        # df = df[['dwv_pw', 'dtv_pw']]
        df = df[['dwv_pw']]
    else:
        df = static_traffic_df[static_traffic_df.intersects(polygon.iloc[0].geometry)]
        if df.empty or df is None:
            return None
        df = df[['DWV_PW']].rename(columns={'DWV_PW': 'dwv_pw'})
    df = df.aggregate(['mean', 'median', 'max', 'count'])
    names = list(product(df.columns, df.index))
    df = df.melt(var_name='traffic type', value_name='value')
    df['traffic type'] = [f'{name[0]} {name[1]}' for name in names]
    df = df.set_index('traffic type').T
    df = df.rename_axis(None, axis=1).reset_index(drop=True)
    return df


def get_pt(point, static_pt_df=None):
    if static_pt_df is None:
        raise NotImplementedError('The following code is not implemented yet')
    else:
        df = static_pt_df[static_pt_df.intersects(point.iloc[0].geometry)]
        if df.empty or df is None:
            print('No public transport data found for the given point {}.'.format(point.iloc[0].geometry))
            return pd.DataFrame(0, columns=['Strasse_Erreichb_EWAP', 'Strasse_Erreichb_EW', 'OeV_Erreichb_EWAP', 'OeV_Erreichb_EW'], index=[0])
        df = df[['Strasse_Erreichb_EWAP', 'Strasse_Erreichb_EW', 'OeV_Erreichb_EWAP', 'OeV_Erreichb_EW']]
    df.reset_index(drop=True, inplace=True)
    return df

def get_population(polygon, static_population_df=None, static_scope_df=None):
    if static_population_df is None:
        df = query_layer(polygon, layer_name='ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner')
        if df.empty or df is None:
            return None
        max_year_census = df.groupby('label').max()['i_year'].reset_index()
        merged_df = pd.merge(df, max_year_census, on=['i_year', 'label'], how='inner')[['number']]
        df = pd.concat([merged_df.sum().rename('sum population'), merged_df.max().rename('max population')], axis=1).reset_index(drop=True)
        raise NotImplementedError('The following code is not implemented yet')
    else:
        nearby_population = static_population_df.loc[static_population_df.within(polygon.iloc[0].geometry), 'NUMMER']
        df_population = pd.DataFrame(data={'sum population': [nearby_population.sum()], 'max_population': [nearby_population.max()]}).fillna(0)

        nearby_scope = static_scope_df.loc[static_scope_df.within(polygon.iloc[0].geometry), ['B08EMPTS2', 'B08EMPTS3']]

        col_names = product(['sum ', 'max '], nearby_scope.columns)
        df_scope = pd.concat([nearby_scope.sum(), nearby_scope.max()], axis=0)
        df_scope = pd.DataFrame(df_scope).T
        df_scope.columns = [t[0]+t[1] for t in col_names]

        # Alternative version, don't know which one is faster
        # df_flat = nearby_scope.agg(['sum', 'max']).stack().reset_index()
        # df_flat['new_index'] = df_flat['level_0'] + ' ' + df_flat['level_1']
        # df_scope = df_flat.set_index('new_index')[0].to_frame().T  # Select the first column and transpose
        # df_scope.columns.name = None

    return pd.concat([df_population, df_scope], axis=1)


def pre_ingestion(query_mapgeoadmin):
    # ---------------------- static population data ----------------------
    static_population_df_url = 'https://data.geo.admin.ch/ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner/volkszaehlung-bevoelkerungsstatistik_einwohner_2021/volkszaehlung-bevoelkerungsstatistik_einwohner_2021_2056.csv'
    if query_mapgeoadmin:
        static_population_df = None
    else:
        static_population_df = pd.read_csv(static_population_df_url, sep=';')
        static_population_df = gpd.GeoDataFrame(static_population_df.drop(columns=['E_KOORD', 'N_KOORD']), geometry=gpd.points_from_xy(static_population_df['E_KOORD'], static_population_df['N_KOORD']), crs='EPSG:2056').to_crs('EPSG:21781')

    # ---------------------- static scope data ----------------------
    static_scope_df_url = 'https://dam-api.bfs.admin.ch/hub/api/dam/assets/27245297/master'
    # download locally and extract the .zip
    if query_mapgeoadmin:
        static_scope_df = None
    else:
        # download locally statisc_scope_url using requests
        response = requests.get(static_scope_df_url)
        with open(join(BASE_PATH, 'static_scope.zip'), 'wb') as f:
            f.write(response.content)
        # extract the .zip
        import zipfile
        dir_path = join(BASE_PATH, 'static_scope', 'ag-b-00.03-22-STATENT2021')
        with zipfile.ZipFile(join(BASE_PATH, 'static_scope.zip'), 'r') as zip_ref:
            zip_ref.extractall(join(BASE_PATH, 'static_scope'))
        # read CSV file STATEN_2021.csv
        static_scope_df = pd.read_csv(join(dir_path, 'STATENT_2021.csv'), sep=';')
        static_scope_df = static_scope_df[['E_KOORD', 'N_KOORD', 'B08EMPTS2', 'B08EMPTS3']]
        static_scope_df = gpd.GeoDataFrame(static_scope_df.drop(columns=['E_KOORD', 'N_KOORD']),
                                           geometry=gpd.points_from_xy(static_scope_df['E_KOORD'],
                                                                       static_scope_df['N_KOORD']),
                                           crs='EPSG:2056').to_crs('EPSG:21781')

    # ---------------------- static traffic data ----------------------
    if query_mapgeoadmin:
        static_traffic_df = None
    else:
        static_traffic_df_local = join(BASE_PATH, 'static_traffic', 'Belastungswerte_Strasse_Schweiz_NPVM_2017+.gpkg')
        if ~exists(static_traffic_df_local):
            os.makedirs(join(BASE_PATH, 'static_traffic'), exist_ok=True)
            static_traffic_df_url = 'https://zenodo.org/records/7649359/files/Belastungswerte_Strasse_Schweiz_NPVM_2017+.gpkg?download=1'
            wget.download(static_traffic_df_url, static_traffic_df_local)
        static_traffic_df = gpd.read_file(static_traffic_df_local, engine='pyogrio', use_arrow=True).to_crs('EPSG:21781')

    # ---------------------- static public transport data ----------------------
    if query_mapgeoadmin:
        static_pt_df = None
    else:
        static_pt_df_local = join(BASE_PATH, 'static_pt', 'erreichbarkeit-oev_2056.gpkg')
        if ~exists(static_pt_df_local):
            os.makedirs(join(BASE_PATH, 'static_pt'), exist_ok=True)
            static_pt_df_url = 'https://data.geo.admin.ch/ch.are.erreichbarkeit-oev/erreichbarkeit-oev/erreichbarkeit-oev_2056.gpkg'
            wget.download(static_pt_df_url, static_pt_df_local)
        static_pt_df = gpd.read_file(static_pt_df_local, engine='pyogrio', use_arrow=True, layer='Reisezeit_Erreichbarkeit').to_crs('EPSG:21781')

    return static_population_df, static_scope_df, static_traffic_df, static_pt_df


def query_stations(stations_lat_long, squared_region=False, radii=(0.5, 1), data_path='.', query_mapgeoadmin=False, previous_df=None):
    cap_style = 3 if squared_region else 1
    static_population_df, static_scope_df, static_traffic_df, static_pt_df = pre_ingestion(query_mapgeoadmin)
    data = []
    for i, lat_long in enumerate(tqdm(stations_lat_long.values)):
        lat, long = lat_long
        #print(' _ point {}: {:.2f}, {:.2f}'.format(i, lat, long))
        point = gpd.GeoDataFrame(geometry=[Point(long, lat)])
        point.crs = 'EPSG:4326'
        point = point.to_crs('EPSG:21781')
        # create a 1 km buffer around the point
        polygons = {k: [] for k in radii}
        for k, v in polygons.items():
            polygon_k = point.copy()
            polygon_k['geometry'] = point.buffer(k*1e3, cap_style=cap_style)
            polygons[k].append(polygon_k)
        temp = []
        for k, v in polygons.items():
            traffic = get_traffic(v[0], static_traffic_df)
            population = get_population(v[0], static_population_df, static_scope_df)
            info = pd.concat([traffic, population], axis=1)
            info.columns = ['{}_{}_km'.format(c, k) for c in info.columns]
            temp.append(info)
        pt = get_pt(point, static_pt_df)
        temp.append(pt)
        data.append(pd.concat(temp, axis=1))
        if (i+1) % 50 == 0:
            data_temp = pd.concat(data)
            data_temp.index = stations_lat_long.index[:i+1]
            if previous_df is not None:
                data_temp = pd.concat([previous_df, data_temp])
            data_temp.to_pickle(join(data_path, 'data_{:05}.zip'.format(data_temp.shape[0])))

    data = pd.concat(data)
    data.index= stations_lat_long.index
    if previous_df is not None:
        data = pd.concat([previous_df, data])
    return pd.concat([stations_lat_long, data], axis=1)


def get_number_of_closest_stations_and_chargers(lat_long, max_dist=1):
    lat_long_gp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lat_long.long, lat_long.lat), crs='EPSG:4326')
    lat_long_gp.to_crs('EPSG:21781', inplace=True)
    lat_long_gp_unique = gpd.GeoDataFrame(geometry=lat_long_gp.geometry.drop_duplicates(), crs='EPSG:21781')
    n_close_points = lat_long_gp.apply(lambda point: np.sum((lat_long_gp_unique.geometry.distance(point.geometry) <= max_dist * 1e3) & (lat_long_gp_unique.geometry.distance(point.geometry) > 0)), axis=1)
    n_close_chargers = lat_long_gp.apply(lambda point: np.sum((lat_long_gp.geometry.distance(point.geometry) <= max_dist * 1e3) & (lat_long_gp.geometry.distance(point.geometry) > 0)), axis=1)
    return pd.DataFrame({'n_close_points_{}_km'.format(max_dist): n_close_points.values,
                         'n_close_chargers_{}_km'.format(max_dist): n_close_chargers.values}, index=lat_long.index)


if __name__ == '__main__':
    pass




