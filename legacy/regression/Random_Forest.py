import os
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import util

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


# Set model construction or inference mode
infer = True
model_number = 4

# Props
db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
prediction_window = 1
street_segment_capacity_limit = 10
data_file_path = "./data/street_segments_occupancy_15min_2019.csv"
weather_data_path = "./data/historical_weather_data_2019-01-01_to_2020-01-01.csv"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
test_start_date = datetime.datetime(2019, 10, 1)
test_end_date = datetime.datetime(2020, 1, 1)

# Random Forrest predictor props
num_predictors = 200
max_depth = 7
min_samples_leaf = 15
max_features = 0.5

if __name__ == '__main__':
    try:
        # make model directory
        if not infer:
            model_number = 1
            model_dir_path = util.make_model_dir("./models/RF", "Random_Forest")
        else:
            model_dir_path = "./models/RF" + "/model-" + str(model_number)

        # Load data
        dataset = util.GenerateOtherMethodsData(data_file_path=data_file_path,
                                                look_back=lookBack, interval=agg_interval,
                                                train_start_date=train_start_date, train_end_date=train_end_date,
                                                test_start_date=test_start_date, test_end_date=test_end_date,
                                                prediction_window=prediction_window,
                                                weather_data_path=None,
                                                capacity_limit=street_segment_capacity_limit, input_3d=False)
        # Training Phase
        if not infer:
            # random_forests = []
            t = time.time()
            # for pred_slot in range(prediction_window):
            #     for i in range(len(dataset.segment_ids)):
            #         logger.debug("Training RF number {}/{} pred window {}"
            #                      .format(i+1, len(dataset.segment_ids), pred_slot + 1))
            #         rf = RandomForestRegressor(n_estimators=num_predictors, max_depth=max_depth,
            #                                    n_jobs=-1, min_samples_leaf=min_samples_leaf, max_features=max_feutures)
            #         rf.fit(dataset.train_X, dataset.train_y[:, i + pred_slot * len(dataset.segment_ids)])
            #         random_forests.append(rf)
            #         with open(model_dir_path + "/RF_region_{}_predslot_{}.pkl".format(i+1, pred_slot + 1), 'wb') as f:
            #             pickle.dump(random_forests[-1], f)
            for i in range(dataset.train_y.shape[1]):
                segment_counter = i % len(dataset.segment_ids) + 1
                pred_slot = i // len(dataset.segment_ids) + 1
                logger.debug(f"Training RF number {segment_counter}/{len(dataset.segment_ids)} pred window {pred_slot}")
                rf = RandomForestRegressor(n_estimators=num_predictors, max_depth=max_depth,
                                           n_jobs=-1, min_samples_leaf=min_samples_leaf, max_features=max_features)
                rf.fit(dataset.train_X, dataset.train_y[:, i])
                # random_forests.append(rf)
                with open(model_dir_path + f"/RF_region_{segment_counter}_predslot_{pred_slot}.pkl", 'wb') as f:
                    pickle.dump(rf, f)
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

        # Inference Phase
        else:
            random_forests = []
            # # Load models
            # for pred_slot in range(prediction_window):
            #     for i in range(len(dataset.segment_ids)):
            #         logger.debug("Loading RF model number {}/{} pred window {}"
            #                      .format(i+1, len(dataset.segment_ids), pred_slot + 1))
            #         with open(model_dir_path + "/RF_region_{}_predslot_{}.pkl".format(i+1, pred_slot + 1), 'rb') as f:
            #             unpickler = pickle.Unpickler(f)
            #             random_forests.append(unpickler.load())
            for i in range(dataset.train_y.shape[1]):
                segment_counter = i % len(dataset.segment_ids) + 1
                pred_slot = i // len(dataset.segment_ids) + 1
                logger.debug(f"Loading RF model number {segment_counter}/{len(dataset.segment_ids)} pred window {pred_slot}")
                with open(model_dir_path + f"/RF_region_{segment_counter}_predslot_{pred_slot}.pkl", 'rb') as f:
                    unpickler = pickle.Unpickler(f)
                    random_forests.append(unpickler.load())
            # Predict
            main_y_hat = np.zeros((dataset.test_X.shape[0], len(dataset.segment_ids) * prediction_window))
            for i in range(len(random_forests)):
                logger.debug("Predicting RF number {}".format(i + 1))
                main_y_hat[:, i] = np.array(random_forests[i].predict(dataset.test_X)).flatten()

            # Add predictions to DataFrame
            dataset.test = util.add_preds(dataset.test, main_y_hat, "RF", dataset.segment_ids)

            # # De-normalize data
            # dataset.test = util.denormalize_data(dataset.test, ['target', 'RF'], dataset.mean, dataset.std)

            # # Round predictions
            # dataset.test = util.round_values(dataset.test, ['RF'])

            # Calculate Errors
            util.calculate_errors(dataset.test, pred="RF")

            # Write results out
            util.result_file(dataset.test, model_path=model_dir_path, model_names=["RF"])

            # # Write predicted data out
            # util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="RF")
    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)
