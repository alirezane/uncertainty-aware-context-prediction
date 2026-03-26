import os
import logging
import pandas as pd

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

input_file_path = './data/processed/street_segments_occupancy_15min_2019.csv'
output_file_path = './data/processed/street_segments_occupancy_15min_2019_cleaned.csv'


if __name__ == '__main__':
    df = pd.read_csv(input_file_path)
    df['timeslot'] = pd.to_datetime(df['timeslot'])
    df.drop_duplicates(inplace=True)
    df = df[df['capacity'] > 0]
    df = df[df['occupancy_ratio'].notna()]
    df = df[(df['occupancy_ratio'] >= 0) & (df['occupancy_ratio'] <= 1)]
    df.sort_values(['streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file_path, index=False)
    logger.debug(f'cleaned occupancy shape: {df.shape}')