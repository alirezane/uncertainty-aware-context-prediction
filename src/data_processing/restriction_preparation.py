import os
import logging
import pandas as pd

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

input_file_path = './data/raw/On-street_Car_Park_Bay_Restrictions.csv'
output_file_path = './data/processed/bay_restrictions_tidy.csv'


def parse_time_only(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x).strftime('%H:%M:%S')
    except BaseException:
        return None


def parse_int(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except BaseException:
        return None


if __name__ == '__main__':
    raw_df = pd.read_csv(input_file_path)
    rows = []
    for _, row in raw_df.iterrows():
        for i in range(1, 7):
            description = row[f'Description{i}'] if f'Description{i}' in raw_df.columns else None
            type_desc = row[f'TypeDesc{i}'] if f'TypeDesc{i}' in raw_df.columns else None
            duration = row[f'Duration{i}'] if f'Duration{i}' in raw_df.columns else None
            if pd.isna(description) and pd.isna(type_desc) and pd.isna(duration):
                continue
            rows.append({
                'BayID': parse_int(row['BayID']) if 'BayID' in raw_df.columns else None,
                'DeviceID': parse_int(row['DeviceID']) if 'DeviceID' in raw_df.columns else None,
                'restriction_number': i,
                'description': None if pd.isna(description) else str(description).strip(),
                'type_desc': None if pd.isna(type_desc) else str(type_desc).strip(),
                'from_day': parse_int(row[f'FromDay{i}']) if f'FromDay{i}' in raw_df.columns else None,
                'to_day': parse_int(row[f'ToDay{i}']) if f'ToDay{i}' in raw_df.columns else None,
                'start_time': parse_time_only(row[f'StartTime{i}']) if f'StartTime{i}' in raw_df.columns else None,
                'end_time': parse_time_only(row[f'EndTime{i}']) if f'EndTime{i}' in raw_df.columns else None,
                'duration_minutes': parse_int(row[f'Duration{i}']) if f'Duration{i}' in raw_df.columns else None,
                'disability_ext_minutes': parse_int(row[f'DisabilityExt{i}']) if f'DisabilityExt{i}' in raw_df.columns else None,
                'effective_on_ph': parse_int(row[f'EffectiveOnPH{i}']) if f'EffectiveOnPH{i}' in raw_df.columns else None,
                'exemption': None if pd.isna(row[f'Exemption{i}']) else str(row[f'Exemption{i}']).strip() if f'Exemption{i}' in raw_df.columns else None
            })
    tidy_df = pd.DataFrame(rows)
    tidy_df.drop_duplicates(inplace=True)
    tidy_df.to_csv(output_file_path, index=False)
    logger.debug(f'tidy restrictions shape: {tidy_df.shape}')