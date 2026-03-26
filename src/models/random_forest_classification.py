import os
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import util_classification

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


infer = True
model_number = 1

db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
prediction_window = 3
street_segment_capacity_limit = 10
data_file_path = "./data/processed/classification_dataset_with_restrictions.csv"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
val_start_date = datetime.datetime(2019, 10, 1)
val_end_date = datetime.datetime(2019, 11, 1)
test_start_date = datetime.datetime(2019, 11, 1)
test_end_date = datetime.datetime(2020, 1, 1)
segments_sample_size = None

num_predictors = 200
max_depth = 7
min_samples_leaf = 15
max_features = 0.5
n_jobs = -1


if __name__ == '__main__':
    try:
        if not infer:
            model_number = 1
            model_dir_path = util_classification.make_model_dir("./models/RF", "random_forest_classification")
        else:
            model_dir_path = "./models/RF/model-" + str(model_number)

        dataset = util_classification.GenerateDate(data_file_path=data_file_path,
                                                   look_back=lookBack,
                                                   train_start_date=train_start_date,
                                                   train_end_date=train_end_date,
                                                   test_start_date=test_start_date,
                                                   test_end_date=test_end_date,
                                                   val_start_date=val_start_date,
                                                   val_end_date=val_end_date,
                                                   prediction_window=prediction_window,
                                                   weather_data_path=None,
                                                   capacity_limit=street_segment_capacity_limit,
                                                   interval=agg_interval,
                                                   normalize_flag=False,
                                                   classification=True,
                                                   one_hot_encoding=False,
                                                   _3d_input=False,
                                                   _3d_output=False,
                                                   random_segments=segments_sample_size,
                                                   use_restrictions=True,
                                                   inject_noise=False,
                                                   noise_std=0.0,
                                                   noise_on_targets=False,
                                                   random_seed=42)

        if not infer:
            logger.debug(f"Training RF model")
            logger.debug(f"num_predictors = {num_predictors}")
            logger.debug(f"max_depth = {max_depth}")
            logger.debug(f"min_samples_leaf = {min_samples_leaf}")
            logger.debug(f"max_features = {max_features}")

            t = time.time()
            rf = MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=num_predictors,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    n_jobs=n_jobs
                ),
                n_jobs=n_jobs
            )
            rf.fit(dataset.train_X, dataset.train_y)

            with open(model_dir_path + "/RF.pkl", 'wb') as f:
                pickle.dump(rf, f)

            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

        else:
            logger.debug("Loading RF model")
            with open(model_dir_path + "/RF.pkl", 'rb') as f:
                rf = pickle.Unpickler(f).load()

            main_y_hat = rf.predict(dataset.test_X)

            dataset.test = util_classification.add_preds(dataset.test, main_y_hat, "RF")

            util_classification.calculate_accuracy(dataset.test,
                                                   dataset.segment_ids,
                                                   prediction_window,
                                                   model_name="RF",
                                                   model_path=model_dir_path)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)