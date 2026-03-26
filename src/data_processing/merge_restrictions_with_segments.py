import os
import logging
import pandas as pd

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

occupancy_file_path = './data/processed/street_segments_occupancy_15min_2019.csv'
restrictions_file_path = './data/processed/street_segments_restrictions_15min_2019.csv'
output_file_path = './data/processed/street_segments_with_restrictions_15min_2019.csv'


if __name__ == '__main__':
    occupancy_df = pd.read_csv(occupancy_file_path)
    restrictions_df = pd.read_csv(restrictions_file_path)
    occupancy_df['timeslot'] = pd.to_datetime(occupancy_df['timeslot'])
    restrictions_df['timeslot'] = pd.to_datetime(restrictions_df['timeslot'])
    merged_df = pd.merge(occupancy_df, restrictions_df,
                         on=['streetid', 'betweenstreet1id', 'betweenstreet2id', 'timeslot'],
                         how='left')
    for col in merged_df.columns:
        if col in ['typedesc_set', 'description_set']:
            merged_df[col] = merged_df[col].fillna('')
        elif str(merged_df[col].dtype) != 'object' and col not in ['timeslot']:
            merged_df[col] = merged_df[col].fillna(0)
    merged_df.to_csv(output_file_path, index=False)
    logger.debug(f'merged segment-restriction shape: {merged_df.shape}')
