import os.path
import time
import logging
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model

import util

pd.options.mode.chained_assignment = None

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s')
logger = logging.getLogger(__name__)

# Model properties
db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
street_segment_capacity_limit = 10
data_file_path = "./data/street_segments_occupancy_15min_2019.csv"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
test_start_date = datetime.datetime(2019, 10, 1)
test_end_date = datetime.datetime(2020, 1, 1)

# make model directory
model_number = 1
model_dir_path = util.make_model_dir("./models/Regression", "Regression")


if __name__ == "__main__":
    # Load data
    dataset = util.GenerateOtherMethodsData(data_file_path=data_file_path,
                                            look_back=lookBack, interval=agg_interval,
                                            train_start_date=train_start_date, train_end_date=train_end_date,
                                            test_start_date=test_start_date, test_end_date=test_end_date,
                                            capacity_limit=street_segment_capacity_limit, input_3d=False)

    # Lasso
    lasso_alpha = 0.1
    lasso_model = linear_model.Lasso(alpha=lasso_alpha)
    logger.debug(f"Fitting LASSO model (alpha={lasso_alpha}):")
    t = time.time()
    lasso_model.fit(dataset.train_X, dataset.train_y)
    lasso_preds = lasso_model.predict(dataset.test_X)
    elapsed = time.time() - t
    logger.debug('LASSO Elapsed: {}'.format(elapsed))

    # OLSR
    olsr_model = linear_model.LinearRegression()
    logger.debug("Fitting OLSR model:")
    t = time.time()
    olsr_model.fit(dataset.train_X, dataset.train_y)
    olsr_preds = olsr_model.predict(dataset.test_X)
    elapsed = time.time() - t
    logger.debug('OLSR  Elapsed: {}'.format(elapsed))

    # Ridge Regression
    ridge_alpha = 0.1
    ridge_solver = 'lsqr'
    ridge_model = linear_model.Ridge(alpha=ridge_alpha, solver=ridge_solver)
    logger.debug(f"Fitting Ridge Regression model: (alpha={ridge_alpha}, solver={ridge_solver})")
    t = time.time()
    ridge_model.fit(dataset.train_X, dataset.train_y)
    ridge_preds = ridge_model.predict(dataset.test_X)
    elapsed = time.time() - t
    logger.debug('Ridge Regression Elapsed: {}'.format(elapsed))

    # Calculate Errors
    util.calculate_errors_raw(dataset.test_y, olsr_preds, model_name="OLSR")
    util.calculate_errors_raw(dataset.test_y, ridge_preds, model_name="Ridge")
    util.calculate_errors_raw(dataset.test_y, lasso_preds, model_name="Lasso")

    # Write results out
    util.result_file_raw(dataset.test_y, [olsr_preds, ridge_preds, lasso_preds],
                         model_path=model_dir_path, model_names=['OLSR', 'Ridge', 'Lasso'])

    # Write predicted data out
    util.predictions_to_csv_raw(lasso_preds, model_dir_path=model_dir_path, model_name="LASSO")
    util.predictions_to_csv_raw(ridge_preds, model_dir_path=model_dir_path, model_name="Ridge")
    util.predictions_to_csv_raw(olsr_preds, model_dir_path=model_dir_path, model_name="OLSR")
