import pandas as pd
from influxdb import InfluxDBClient, DataFrameClient
import urllib3
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from absl import app
from absl import flags
from os.path import join, exists
from os import makedirs


urllib3.disable_warnings()


flags.DEFINE_string('save_dir', 'data', 'directory in which to save the raw datasets')
flags.DEFINE_string('dt', '15m', 'Sampling time')

flags.DEFINE_string('t_start', '2024-05-14', 'Download start time')
flags.DEFINE_string('t_end', '2024-05-15', 'Download stop time')

flags.DEFINE_string('db_conf_path', 'conf/dbs_conf.json', 'configuration file for the database')
# BBOX Swiss coordinates
# lat_max, lon_max = 47.808464, 10.492294
# lat_min, lon_min = 45.817920, 5.956085
flags.DEFINE_float('lat_min', 46.00, 'latitude min')
flags.DEFINE_float('lat_max', 46.03, 'latitude max')
flags.DEFINE_float('lon_min', 8.9, 'longitude min')
flags.DEFINE_float('lon_max', 8.96, 'longitude max')

flags.DEFINE_boolean('raw', False, 'If true, this will download the status codes. If false, '
                                   'it will download the number of times the status was 0 or 1 in the time interval dt '
                                   '(available and occupied, respectively). WARNING: If raw=TRUE this must be call at '
                                   'the maximum frequency of the data dt = 1m otherwise you\'ll get averages of the '
                                   'status (status codes are 0=AVAILABLE 1=OCCUPIED -1=NOT_IDENTIFIABLE -2=UNKNOWN '
                                   '-3=OUTOFSERVICE')
FLAGS = flags.FLAGS


def do_query(qy, dfc):
    try:
        res_df = dfc.query(qy)
    except Exception as e:
        print('EXCEPTION: %s' % str(e))
        return None
    return res_df


def replace_key(key):
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], tuple):
        return key[1][0][1]  # Extract the value from the nested tuple
    else:
        return key


def results_to_df(results):
    new_dict = dict(map(lambda item: (replace_key(item[0]), item[1]), results.items()))
    combined_df = pd.concat([df.assign(key=key) for key, df in new_dict.items()], ignore_index=False).reset_index(drop=False)
    combined_df.rename(columns={'index': 'time'}, inplace=True)
    return combined_df


def query_raw(df_client, db_conf, lat_min, lat_max, lon_min, lon_max, t_start, t_end, dt='1m'):
    """
    This must be call at the maximum frequency of the data = 1 minute, otherwise you'll get averages of the status
    :param df_client:
    :param db_conf:
    :param lat_min:
    :param lat_max:
    :param lon_min:
    :param lon_max:
    :param t_start:
    :param t_end:
    :param state: 0=AVAILABLE 1=OCCUPIED -1=NOT_IDENTIFIABLE -2=UNKNOWN -3=OUTOFSERVICE
    :param dt:
    :return:
    """

    # query one day at a time
    results_list = []
    for t in tqdm(pd.date_range(t_start, t_end, freq='D'), desc='Downloading data day by day'):
        t_start = t.tz_localize(None).isoformat() + "Z"
        t_end = (t + pd.to_timedelta('1d')).tz_localize(None).isoformat() + "Z"

        query = (f"SELECT MEAN(value) "
                 f"FROM {db_conf['measurement']} "
                 f"WHERE time>='{t_start}' "
                 f"AND time<'{t_end}' "
                 f"AND latitude>={lat_min} "
                 f"AND latitude<={lat_max} "
                 f"AND longitude>={lon_min} "
                 f"AND longitude<={lon_max} "
                 f"GROUP BY time({dt}), evse_id"
                 )

        results = do_query(query, df_client)
        if results is None or results == {}:
            continue
        else:
            results_list.append(results_to_df(results))

    df = pd.concat(results_list)
    df.reset_index(drop=True, inplace=True)
    df['time'] = pd.Index(df['time']).tz_convert('Europe/Zurich')

    return df


def query_status(df_client, db_conf, lat_min, lat_max, lon_min, lon_max, t_start, t_end, dt='15m', state=0):
    """
    This returns number of times the status was state in the time interval dt
    :param df_client: DataFrameClient
    :param db_conf: dict
    :param lat_min:
    :param lat_max:
    :param lon_min:
    :param lon_max:
    :param t_start:
    :param t_end:
    :param state: 0=AVAILABLE 1=OCCUPIED -1=NOT_IDENTIFIABLE -2=UNKNOWN -3=OUTOFSERVICE
    :param dt:
    :return:
    """

    # query one day at a time
    results_list = []
    for t in tqdm(pd.date_range(t_start, t_end, freq='D'),
                  desc='Downloading data day by day, for state {}'.format(state)):
        t_start = t.tz_localize(None).isoformat() + "Z"
        t_end = (t + pd.to_timedelta('1d')).tz_localize(None).isoformat() + "Z"

        query = (f"SELECT COUNT(value) AS count "
                 f"FROM {db_conf['measurement']} "
                 f"WHERE time>='{t_start}' "
                 f"AND time<'{t_end}' "
                 f"AND latitude>={lat_min} "
                 f"AND latitude<={lat_max} "
                 f"AND longitude>={lon_min} "
                 f"AND longitude<={lon_max} "
                 f"AND value = {state} "
                 f"GROUP BY time({dt}), evse_id"
                 )

        results = do_query(query, df_client)
        if results is None or results == {}:
            continue
        else:
            results_list.append(results_to_df(results))

    df = pd.concat(results_list)
    df.reset_index(drop=True, inplace=True)
    df['time'] = pd.Index(df['time']).tz_convert('Europe/Zurich')

    return df


def get_client(cfg_path=None, db_conf=None):
    if db_conf is None:
        if cfg_path is None:
            cfg_path = 'conf/dbs_conf.json'
        with open(cfg_path) as f:
            db_conf = json.load(f)['evs']
    client = DataFrameClient(host=db_conf['host'], port=db_conf['port'], password=db_conf['password'],
                    username=db_conf['user'], database=db_conf['database'], ssl=db_conf['ssl'])
    return client, db_conf


def get_data(dt, t_start, t_end, lat_min, lat_max, lon_min, lon_max, raw, save_data=True, save_dir='./data',
             db_conf=None, db_conf_path='conf/dbs_conf.json'):
    """
    :param dt: time interval
    :param t_start: start time
    :param t_end: end time
    :param lat_min: minimum latitude
    :param lat_max: maximum latitude
    :param lon_min: minimum longitude
    :param lon_max: maximum longitude
    :param raw: If true, this will download the status codes. If false, it will download the number of times the status
                was 0 or 1 in the time interval dt (available and occupied, respectively). WARNING: If raw=TRUE this
                must be call at the maximum frequency of the data dt = 1m otherwise you\'ll get averages of the status
                (status codes are 0=AVAILABLE 1=OCCUPIED -1=NOT_IDENTIFIABLE -2=UNKNOWN -3=OUTOFSERVICE
    :param save_data:  If true, save the data to a pickle file
    :param save_dir:  directory in which to save the raw datasets
    :param db_conf:  dict with the database configuration, if not None it will be used instead of the db_conf_path
    :param db_conf_path:  path to the database configuration file
    :return:
    """
    df_client, db_conf = get_client(cfg_path=db_conf_path, db_conf=db_conf)
    t_start = pd.Timestamp(t_start).isoformat() + "Z"
    t_end = pd.Timestamp(t_end).isoformat() + "Z"
    raw = raw

    if raw:
        if dt != '1m':
            ValueError(
                'This function must be called at the maximum frequency of the data = 1 minute, otherwise '
                'you\'ll get averages of the status')
        data = query_raw(df_client, db_conf, lat_min, lat_max, lon_min, lon_max,
                         t_start, t_end, dt=dt)
    else:
        available = query_status(df_client, db_conf, lat_min, lat_max, lon_min, lon_max,
                                 t_start, t_end, dt=dt, state=0)
        occupied = query_status(df_client, db_conf, lat_min, lat_max, lon_min, lon_max,
                                 t_start, t_end, dt=dt, state=1)
        available.rename(columns={'count': 'available'}, inplace=True)
        occupied.rename(columns={'count': 'occupied'}, inplace=True)

        data = pd.merge(available, occupied, on=['time', 'key'], how='outer')

        data[['occupied', 'available']] = data[['occupied', 'available']].fillna(0)
        data[['occupied', 'available']] = data[['occupied', 'available']].astype(int)

    if save_data:
        pars = {'lat_min': lat_min, 'lat_max': lat_max, 'lon_min': lon_min, 'lon_max': lon_max,
                't_start': t_start, 't_end': t_end, 'dt': dt, 'raw': raw}
        # if save_dir doesn't exist, create it
        if not exists(save_dir):
            makedirs(save_dir)
        now = pd.Timestamp.now()
        data.to_pickle(join(save_dir, 'ev_occupancy_{}.zip'.format(now.strftime('%Y-%m-%d--%H-%M-%S'))))
        pd.DataFrame(pars, index=[0]).to_pickle(join(save_dir, 'pars_{}.pk'.format(now.strftime('%Y-%m-%d--%H-%M-%S'))))
    return data


def plot_sample(data, n_max=1000):
    raw = not 'available' in data.columns
    if raw:
        plt.figure(figsize=(15, 6))
        for n in np.unique(data['key']):
            plt.plot(data.loc[data['key'] == n, ['time', 'mean']].set_index("time").iloc[:n_max])
    else:
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), layout='tight')
        for n in np.unique(data['key']):
            ax[0].plot(data.loc[data['key'] == n, ['time', 'available']].set_index("time").iloc[:n_max])
            ax[1].plot(data.loc[data['key'] == n, ['time', 'occupied']].set_index("time").iloc[:n_max])
        ax[0].set_title('Available')
        ax[1].set_title('Occupied')
    plt.show()


def main(unused_argv) -> None:
    data = get_data(FLAGS.dt, FLAGS.t_start, FLAGS.t_end, FLAGS.lat_min, FLAGS.lat_max, FLAGS.lon_min, FLAGS.lon_max,
                    FLAGS.raw, save_data=True, save_dir=FLAGS.save_dir)
    plot_sample(data)


if __name__ == '__main__':
    app.run(main)