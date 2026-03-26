# create_database.py
import os
import logging
import datetime
import numpy as np
import pandas as pd
import sqlalchemy as sa
from config.db_config import db_user, db_password
from src.data_collection.segments_location_data import segments_location_df

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


db_name = "melbourne-parking"
year = 2019
chunk_size = 10 ** 6
aggregation_interval = 15
sensor_data_file_path = "./data/raw/On-street_Car_Parking_Sensor_Data_-_2019.csv"
restrictions_data_file_path = "./data/raw/On-street_Car_Park_Bay_Restrictions.csv"
engine = sa.create_engine(f'postgresql://{db_user}:{db_password}@localhost:5432/{db_name}')


public_holidays = [
    datetime.date(2019, 1, 1), datetime.date(2019, 1, 28), datetime.date(2019, 3, 11),
    datetime.date(2019, 4, 19), datetime.date(2019, 4, 20), datetime.date(2019, 4, 21),
    datetime.date(2019, 4, 22), datetime.date(2019, 4, 25), datetime.date(2019, 6, 10),
    datetime.date(2019, 9, 27), datetime.date(2019, 11, 5), datetime.date(2019, 12, 25),
    datetime.date(2019, 12, 26)
]


def parse_bool(x):
    if pd.isna(x):
        return None
    try:
        return bool(x)
    except BaseException:
        x = str(x).strip().lower()
        if x in ['true', 't', '1', 'yes', 'y']:
            return True
        if x in ['false', 'f', '0', 'no', 'n']:
            return False
        return None


def parse_time_only(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x).time()
    except BaseException:
        return None


def parse_int(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except BaseException:
        return None


def flatten_restrictions(raw_df):
    rows = []
    for _, row in raw_df.iterrows():
        bay_id = parse_int(row['BayID']) if 'BayID' in raw_df.columns else None
        device_id = parse_int(row['DeviceID']) if 'DeviceID' in raw_df.columns else None
        for i in range(1, 7):
            description = row[f'Description{i}'] if f'Description{i}' in raw_df.columns else None
            disability_ext = row[f'DisabilityExt{i}'] if f'DisabilityExt{i}' in raw_df.columns else None
            duration = row[f'Duration{i}'] if f'Duration{i}' in raw_df.columns else None
            effective_on_ph = row[f'EffectiveOnPH{i}'] if f'EffectiveOnPH{i}' in raw_df.columns else None
            end_time = row[f'EndTime{i}'] if f'EndTime{i}' in raw_df.columns else None
            exemption = row[f'Exemption{i}'] if f'Exemption{i}' in raw_df.columns else None
            from_day = row[f'FromDay{i}'] if f'FromDay{i}' in raw_df.columns else None
            start_time = row[f'StartTime{i}'] if f'StartTime{i}' in raw_df.columns else None
            to_day = row[f'ToDay{i}'] if f'ToDay{i}' in raw_df.columns else None
            type_desc = row[f'TypeDesc{i}'] if f'TypeDesc{i}' in raw_df.columns else None
            if pd.isna(description) and pd.isna(type_desc) and pd.isna(duration) and pd.isna(from_day) and pd.isna(to_day):
                continue
            rows.append({
                'bayid': bay_id,
                'deviceid': device_id,
                'restriction_number': i,
                'description': None if pd.isna(description) else str(description).strip(),
                'typedesc': None if pd.isna(type_desc) else str(type_desc).strip(),
                'fromday': parse_int(from_day),
                'today': parse_int(to_day),
                'starttime': parse_time_only(start_time),
                'endtime': parse_time_only(end_time),
                'duration_minutes': parse_int(duration),
                'disabilityext_minutes': parse_int(disability_ext),
                'effectiveonph': parse_int(effective_on_ph),
                'exemption': None if pd.isna(exemption) else str(exemption).strip()
            })
    restrictions = pd.DataFrame(rows)
    restrictions.drop_duplicates(inplace=True)
    restrictions.reset_index(drop=True, inplace=True)
    return restrictions


def create_raw_parking_table():
    table_name = f'parking_sensor_events_{year}'
    chunk_reader = pd.read_csv(filepath_or_buffer=sensor_data_file_path, chunksize=chunk_size)
    first_chunk = True
    chunk_counter = 0
    for chunk in chunk_reader:
        chunk['ArrivalTime'] = pd.to_datetime(chunk['ArrivalTime'])
        chunk['DepartureTime'] = pd.to_datetime(chunk['DepartureTime'])
        chunk.columns = [x.lower() for x in chunk.columns]
        chunk.to_sql(table_name, engine, if_exists='replace' if first_chunk else 'append', index=False)
        first_chunk = False
        chunk_counter += 1
        logger.debug(f'raw parking chunk {chunk_counter} inserted')


def create_bay_restrictions_table():
    table_name = 'bay_restrictions_2019'
    raw_df = pd.read_csv(restrictions_data_file_path)
    tidy_df = flatten_restrictions(raw_df)
    tidy_df.to_sql(table_name, engine, if_exists='replace', index=False)
    logger.debug(f'bay restrictions table created with shape {tidy_df.shape}')


def create_bays_table():
    table_name = f'parking_bays_{year}'
    cmd = f'''
        CREATE TABLE {table_name} AS
        SELECT bayid,
               deviceid,
               streetid,
               streetname,
               betweenstreet1id,
               betweenstreet1,
               betweenstreet2id,
               betweenstreet2,
               sidename,
               areaname,
               signplateid,
               sign,
               MIN(arrivaltime) AS first_seen,
               MAX(departuretime) AS last_seen
        FROM parking_sensor_events_{year}
        GROUP BY bayid, deviceid, streetid, streetname, betweenstreet1id, betweenstreet1,
                 betweenstreet2id, betweenstreet2, sidename, areaname, signplateid, sign
    '''
    with engine.connect() as conn:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name}'))
        conn.execute(sa.text(cmd))
        conn.commit()


def create_occupancy_table(interval=15):
    bays_cmd = f'''SELECT DISTINCT bayid, streetid, betweenstreet1id, betweenstreet2id
                   FROM parking_sensor_events_{year}
                   WHERE bayid IS NOT NULL'''
    bays_df = pd.read_sql(bays_cmd, engine)
    occupancy_rows = []
    table_name = f'parking_bays_occupancy_{interval}min_{year}'

    for i in range(bays_df.shape[0]):
        bay = bays_df.iloc[i]
        bay_id = int(bay['bayid'])
        streetid = int(bay['streetid']) if pd.notna(bay['streetid']) else None
        bs1 = int(bay['betweenstreet1id']) if pd.notna(bay['betweenstreet1id']) else None
        bs2 = int(bay['betweenstreet2id']) if pd.notna(bay['betweenstreet2id']) else None

        bay_events_cmd = f'''
            SELECT arrivaltime, departuretime
            FROM parking_sensor_events_{year}
            WHERE bayid = {bay_id}
            ORDER BY arrivaltime
        '''
        bay_events = pd.read_sql(bay_events_cmd, engine)
        if bay_events.shape[0] == 0:
            continue

        bay_events['arrivaltime'] = pd.to_datetime(bay_events['arrivaltime'])
        bay_events['departuretime'] = pd.to_datetime(bay_events['departuretime'])

        slot_start = datetime.datetime(year, 1, 1, 0, 0)
        slot_end = datetime.datetime(year + 1, 1, 1, 0, 0)
        slots = pd.date_range(slot_start, slot_end, freq=f'{interval}min', inclusive='left')

        for slot in slots:
            slot_finish = slot + datetime.timedelta(minutes=interval)
            present = ((bay_events['arrivaltime'] < slot_finish) & (bay_events['departuretime'] > slot)).any()
            occupancy_rows.append({
                'bayid': bay_id,
                'streetid': streetid,
                'betweenstreet1id': bs1,
                'betweenstreet2id': bs2,
                'timeslot': slot,
                'vehiclepresent': bool(present)
            })

        logger.debug(f'bay {i + 1}/{bays_df.shape[0]} processed for occupancy')

    occupancy_df = pd.DataFrame(occupancy_rows)
    occupancy_df.to_sql(table_name, engine, if_exists='replace', index=False)
    logger.debug(f'bay occupancy table created with shape {occupancy_df.shape}')


def create_street_segments_occupancy(interval=15):
    bay_table = f'parking_bays_occupancy_{interval}min_{year}'
    segment_table = f'street_segments_occupancy_{interval}min_{year}'
    cmd = f'''
        CREATE TABLE {segment_table} AS
        SELECT streetid,
               betweenstreet1id,
               betweenstreet2id,
               timeslot,
               COUNT(*) AS capacity,
               COUNT(CASE WHEN vehiclepresent THEN 1 END) AS occupied,
               CAST(COUNT(CASE WHEN vehiclepresent THEN 1 END) AS FLOAT) / COUNT(*) AS occupancy_ratio
        FROM {bay_table}
        GROUP BY streetid, betweenstreet1id, betweenstreet2id, timeslot
        ORDER BY streetid, betweenstreet1id, betweenstreet2id, timeslot
    '''
    with engine.connect() as conn:
        conn.execute(sa.text(f'DROP TABLE IF EXISTS {segment_table}'))
        conn.execute(sa.text(cmd))
        conn.commit()


def create_segment_metadata(interval=15):
    table_name = f'street_segments_{year}'
    bays_cmd = f'''
        SELECT DISTINCT bayid, streetid, betweenstreet1id, betweenstreet2id, streetname, betweenstreet1, betweenstreet2
        FROM parking_sensor_events_{year}
        WHERE bayid IS NOT NULL
    '''
    bays_df = pd.read_sql(bays_cmd, engine)
    bays_df.drop_duplicates(subset=['bayid'], inplace=True)

    if segments_location_df is not None and segments_location_df.shape[0] > 0:
        loc = segments_location_df.copy()
        loc.columns = [x.lower() for x in loc.columns]
        bays_df['streetname_upper'] = bays_df['streetname'].astype(str).str.upper()
        bays_df['betweenstreet1_upper'] = bays_df['betweenstreet1'].astype(str).str.upper()
        bays_df['betweenstreet2_upper'] = bays_df['betweenstreet2'].astype(str).str.upper()
        loc['streetname'] = loc['streetname'].astype(str).str.upper()
        loc['betweenstreet1'] = loc['betweenstreet1'].astype(str).str.upper()
        loc['betweenstreet2'] = loc['betweenstreet2'].astype(str).str.upper()
        bays_df = pd.merge(bays_df, loc,
                           left_on=['streetname_upper', 'betweenstreet1_upper', 'betweenstreet2_upper'],
                           right_on=['streetname', 'betweenstreet1', 'betweenstreet2'],
                           how='left')

    seg_df = bays_df.groupby(['streetid', 'betweenstreet1id', 'betweenstreet2id'], as_index=False).agg({
        'bayid': 'count',
        'streetname': 'first',
        'betweenstreet1': 'first',
        'betweenstreet2': 'first',
        'latitude': 'mean' if 'latitude' in bays_df.columns else 'first',
        'longitude': 'mean' if 'longitude' in bays_df.columns else 'first'
    })
    seg_df.rename(columns={'bayid': 'capacity'}, inplace=True)
    seg_df.to_sql(table_name, engine, if_exists='replace', index=False)


def restriction_active(row, slot_dt):
    if pd.isna(row['fromday']) or pd.isna(row['today']) or pd.isna(row['starttime']) or pd.isna(row['endtime']):
        return False
    day_of_week = (slot_dt.weekday() + 1) % 7
    from_day = int(row['fromday'])
    to_day = int(row['today'])
    if from_day <= to_day:
        valid_day = from_day <= day_of_week <= to_day
    else:
        valid_day = day_of_week >= from_day or day_of_week <= to_day
    if not valid_day:
        return False
    if slot_dt.date() in public_holidays and pd.notna(row['effectiveonph']) and int(row['effectiveonph']) == 0:
        return False
    slot_time = slot_dt.time()
    start_time = row['starttime']
    end_time = row['endtime']
    if start_time <= end_time:
        return start_time <= slot_time < end_time
    return slot_time >= start_time or slot_time < end_time


def create_segment_restrictions(interval=15):
    restrictions = pd.read_sql('SELECT * FROM bay_restrictions_2019', engine)
    bays = pd.read_sql(f'''
        SELECT DISTINCT bayid, streetid, betweenstreet1id, betweenstreet2id
        FROM parking_sensor_events_{year}
        WHERE bayid IS NOT NULL
    ''', engine)
    bays.drop_duplicates(subset=['bayid'], inplace=True)
    restrictions = pd.merge(restrictions, bays, on='bayid', how='inner')

    rows = []
    slots = pd.date_range(datetime.datetime(year, 1, 1, 0, 0),
                          datetime.datetime(year + 1, 1, 1, 0, 0),
                          freq=f'{interval}min', inclusive='left')

    for bay_id, bay_df in restrictions.groupby('bayid'):
        for slot in slots:
            active = bay_df[bay_df.apply(lambda x: restriction_active(x, slot), axis=1)]
            if active.shape[0] == 0:
                continue
            for _, r in active.iterrows():
                rows.append({
                    'bayid': int(r['bayid']),
                    'streetid': int(r['streetid']),
                    'betweenstreet1id': int(r['betweenstreet1id']),
                    'betweenstreet2id': int(r['betweenstreet2id']),
                    'timeslot': slot,
                    'restriction_active': 1,
                    'typedesc': r['typedesc'],
                    'description': r['description'],
                    'duration_minutes': r['duration_minutes'],
                    'effectiveonph': r['effectiveonph'],
                    'exemption': r['exemption'],
                    'disabilityext_minutes': r['disabilityext_minutes']
                })
        logger.debug(f'bay {bay_id} processed for restriction timeslots')

    active_df = pd.DataFrame(rows)
    active_df.to_sql(f'parking_bays_restrictions_timeslot_{interval}min_{year}', engine, if_exists='replace', index=False)

    if active_df.shape[0] == 0:
        return

    active_df['typedesc_clean'] = active_df['typedesc'].fillna('').astype(str).str.lower()
    active_df['description_clean'] = active_df['description'].fillna('').astype(str).str.lower()
    active_df['is_meter'] = active_df['typedesc_clean'].str.contains('meter').astype(int)
    active_df['is_disabled'] = (active_df['typedesc_clean'].str.contains('disabled') |
                                active_df['description_clean'].str.contains('dis')).astype(int)
    active_df['is_loading'] = (active_df['typedesc_clean'].str.contains('loading') |
                               active_df['description_clean'].str.contains('loading')).astype(int)
    active_df['is_permit'] = (active_df['typedesc_clean'].str.contains('permit') |
                              active_df['description_clean'].str.contains('permit')).astype(int)
    active_df['is_clearway'] = (active_df['typedesc_clean'].str.contains('clearway') |
                                active_df['description_clean'].str.contains('clearway')).astype(int)
    active_df['is_no_parking'] = (active_df['typedesc_clean'].str.contains('no parking') |
                                  active_df['description_clean'].str.contains('no parking')).astype(int)
    active_df['is_no_stopping'] = (active_df['typedesc_clean'].str.contains('no stopping') |
                                   active_df['description_clean'].str.contains('no stopping')).astype(int)

    segment_group = active_df.groupby(['streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot'], as_index=False).agg({
        'bayid': 'count',
        'restriction_active': 'sum',
        'duration_minutes': ['mean', 'min', 'max'],
        'disabilityext_minutes': ['mean', 'min', 'max'],
        'effectiveonph': 'max',
        'is_meter': 'mean',
        'is_disabled': 'mean',
        'is_loading': 'mean',
        'is_permit': 'mean',
        'is_clearway': 'mean',
        'is_no_parking': 'mean',
        'is_no_stopping': 'mean',
        'typedesc': lambda x: '|'.join(sorted(list(set([str(v) for v in x.dropna() if str(v).strip() != ''])))),
        'description': lambda x: '|'.join(sorted(list(set([str(v) for v in x.dropna() if str(v).strip() != '']))))
    })

    segment_group.columns = [
        'streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot',
        'restricted_bay_count', 'active_restriction_count',
        'duration_mean', 'duration_min', 'duration_max',
        'disability_ext_mean', 'disability_ext_min', 'disability_ext_max',
        'effectiveonph', 'meter_ratio', 'disabled_ratio', 'loading_ratio', 'permit_ratio',
        'clearway_ratio', 'no_parking_ratio', 'no_stopping_ratio',
        'typedesc_set', 'description_set'
    ]

    capacity_df = pd.read_sql(f'''
        SELECT streetid, betweenstreet1id, betweenstreet2id, timeslot, capacity
        FROM street_segments_occupancy_{interval}min_{year}
    ''', engine)
    segment_group = pd.merge(capacity_df, segment_group,
                             on=['streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot'], how='left')
    segment_group['restricted_bay_count'] = segment_group['restricted_bay_count'].fillna(0)
    segment_group['active_restriction_ratio'] = segment_group['restricted_bay_count'] / segment_group['capacity']
    segment_group.to_sql(f'street_segments_restrictions_{interval}min_{year}', engine, if_exists='replace', index=False)


def export_processed_csvs(interval=15):
    occupancy = pd.read_sql(f'SELECT * FROM street_segments_occupancy_{interval}min_{year}', engine)
    occupancy.to_csv(f'./data/processed/street_segments_occupancy_{interval}min_{year}.csv', index=False)

    restrictions = pd.read_sql(f'SELECT * FROM street_segments_restrictions_{interval}min_{year}', engine)
    restrictions.to_csv(f'./data/processed/street_segments_restrictions_{interval}min_{year}.csv', index=False)


if __name__ == '__main__':
    create_raw_parking_table()
    create_bay_restrictions_table()
    create_bays_table()
    create_occupancy_table(aggregation_interval)
    create_street_segments_occupancy(aggregation_interval)
    create_segment_metadata(aggregation_interval)
    create_segment_restrictions(aggregation_interval)
    export_processed_csvs(aggregation_interval)


