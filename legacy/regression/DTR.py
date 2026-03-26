import gc
import os
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
# import sqlalchemy as sa
from sklearn.tree import DecisionTreeRegressor

import util
# from config.db_config import db_user, db_password

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None
os.environ['NUMEXPR_MAX_THREADS'] = '30'


# Set model construction or inference mode
infer = True
model_number = 1

# Props
db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
street_segment_capacity_limit = 10
data_file_path = "./data/street_segments_occupancy_15min_2019.csv"
# engine = sa.create_engine(f'postgresql://{db_user}:{db_password}@localhost:5432/{db_name}')
# street_segments_occupancy_table_name = "street_segments_occupancy_5min_2019"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
test_start_date = datetime.datetime(2019, 10, 1)
test_end_date = datetime.datetime(2020, 1, 1)


if __name__ == '__main__':
    try:
        # make model directory
        if not infer:
            model_number = 1
            model_dir_path = util.make_model_dir("./models/DTR", "DTR")
        else:
            model_dir_path = "./models/DTR/model-" + str(model_number)

        # Load data
        dataset = util.GenerateOtherMethodsData(data_file_path=data_file_path,
                                                look_back=lookBack, interval=agg_interval,
                                                train_start_date=train_start_date, train_end_date=train_end_date,
                                                test_start_date=test_start_date, test_end_date=test_end_date,
                                                capacity_limit=street_segment_capacity_limit, input_3d=False)
        # Training Phase
        if not infer:
            t = time.time()
            dtrs = []
            for i in range(len(dataset.segment_ids)):
                logger.debug("Training DTR model for segment number {}".format(i+1))
                dtr = DecisionTreeRegressor()
                dtr.fit(dataset.train_X, dataset.train_y[:, i])
                dtrs.append(dtr)
                with open(model_dir_path + "/DTR_segment_{}.pkl".format(i+1), 'wb') as f:
                    pickle.dump(dtrs[-1], f)
                gc.collect()
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

        # Inference Phase
        else:
            dtrs = []
            # Load models
            for i in range(len(dataset.segment_ids)):
                with open(model_dir_path + "/DTR_segment_{}.pkl".format(i+1), 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    dtrs.append(unpickler.load())
            # Predict
            main_y_hat = np.zeros((dataset.test_X.shape[0], len(dataset.segment_ids)))
            for i in range(len(dtrs)):
                logger.debug("Predicting DTR model for segment number {}".format(i + 1))
                main_y_hat[:, i] = np.array(dtrs[i].predict(dataset.test_X)).flatten()

            # Add predictions to DataFrame
            dataset.test = util.add_preds(dataset.test, main_y_hat, "DTR", dataset.segment_ids)

            # # Round predictions
            # dataset.test = util.round_values(dataset.test, ['DTR'])

            # Calculate Errors
            util.calculate_errors(dataset.test, pred="DTR")

            # Write results out
            util.result_file(dataset.test, model_path=model_dir_path, model_names=["DTR"])

            # # Write predicted data out
            # util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="DTR")
    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)
