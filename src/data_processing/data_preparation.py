import os
import logging
import datetime
import numpy as np
import pandas as pd
from config.occupancy_classes import occupancy_classes

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

occupancy_file_path = './data/processed/street_segments_occupancy_15min_2019_cleaned.csv'
weather_file_path = './data/external/historical_weather_data_2019-01-01_to_2020-01-01.csv'
restrictions_file_path = './data/processed/street_segments_restrictions_15min_2019.csv'
output_dir = './data/processed'
look_back = 8
prediction_window = 3
use_weather = True
use_restrictions = True


def ratio_to_class(x):
    if pd.isna(x):
        return None
    for class_id, class_info in occupancy_classes.items():
        if x >= class_info['min'] and x <= class_info['max']:
            if x == 1 and class_id != 4:
                continue
            return class_id
    if x == 1:
        return 4
    return None


def prepare_weather(df):
    weather_df = pd.read_csv(weather_file_path)
    weather_df['DateTime'] = pd.to_datetime(weather_df['DateTime'])
    weather_df.columns = [x.replace(' ', '_') for x in weather_df.columns]
    cols = [x for x in weather_df.columns if x != 'DateTime']
    for col in cols:
        weather_df[col] = weather_df[col].fillna(method='ffill')
    if 'Temperature' in weather_df.columns:
        weather_df['Temperature'] = weather_df['Temperature'].astype(str).str.extract('(\d+)')[0].astype(float)
    if 'Humidity' in weather_df.columns:
        weather_df['Humidity'] = weather_df['Humidity'].astype(str).str.extract('(\d+)')[0].astype(float)
    if 'Wind_Speed' in weather_df.columns:
        weather_df['Wind_Speed'] = weather_df['Wind_Speed'].astype(str).str.extract('(\d+)')[0].astype(float)
    if 'Condition' in weather_df.columns:
        weather_df['Condition'] = weather_df['Condition'].astype(str).str.replace(' / ', '/', regex=False).str.replace(' ', '', regex=False)
        weather_df = pd.concat([weather_df, pd.get_dummies(weather_df['Condition'], prefix='weather_condition')], axis=1)
        weather_df.drop(['Condition'], axis=1, inplace=True)
    for col in [x for x in weather_df.columns if x != 'DateTime']:
        if str(weather_df[col].dtype) != 'uint8':
            max_val = weather_df[col].max()
            min_val = weather_df[col].min()
            if pd.notna(max_val) and pd.notna(min_val) and max_val != min_val:
                weather_df[col] = (weather_df[col] - min_val) / (max_val - min_val)
    weather_df.rename(columns={'DateTime': 'timeslot'}, inplace=True)
    return pd.merge(df, weather_df, on='timeslot', how='left')


def prepare_restrictions(df):
    restrictions_df = pd.read_csv(restrictions_file_path)
    restrictions_df['timeslot'] = pd.to_datetime(restrictions_df['timeslot'])
    merge_cols = ['streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot']
    restrictions_cols = [x for x in restrictions_df.columns if x not in merge_cols]
    logger.debug(f'prepared dataset shape: {prepared_df.shape}')