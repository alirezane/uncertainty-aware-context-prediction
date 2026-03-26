import os
import time
import pickle
import logging
import datetime
import pandas as pd
from sklearn.svm import SVC
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

kernel = "rbf"
C = 1.0
gamma = "scale"


if __name__ == '__main__':
    try:
        if not infer:
            model_number = 1
            model_dir_path = util_classification.make_model_dir("./models/SVM", "svm_classification")
        else:
            model_dir_path = "./models/SVM/model-" + str(model_number)

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
            logger.debug(f"Training SVM model")
            logger.debug(f"kernel = {kernel}")
            logger.debug(f"C = {C}")
            logger.debug(f"gamma = {gamma}")

            t = time.time()
            svm = MultiOutputClassifier(
                SVC(kernel=kernel,
                    C=C,
                    gamma=gamma)
            )
            svm.fit(dataset.train_X, dataset.train_y)

            with open(model_dir_path + "/SVM.pkl", 'wb') as f:
                pickle.dump(svm, f)

            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

        else:
            logger.debug("Loading SVM model")
            with open(model_dir_path + "/SVM.pkl", 'rb') as f:
                svm = pickle.Unpickler(f).load()

            main_y_hat = svm.predict(dataset.test_X)

            dataset.test = util_classification.add_preds(dataset.test, main_y_hat, "SVM")

            util_classification.calculate_accuracy(dataset.test,
                                                   dataset.segment_ids,
                                                   prediction_window,
                                                   model_name="SVM",
                                                   model_path=model_dir_path)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)