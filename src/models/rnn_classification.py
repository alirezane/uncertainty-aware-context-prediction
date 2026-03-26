import os
import time
import logging
import datetime
import pandas as pd

import util_classification

model_type = "LSTM"

log_file_path = './logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower() + f'_{model_type.lower()}')
logging.basicConfig(filename=log_file_path,
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None
os.environ['NUMEXPR_MAX_THREADS'] = '30'


infer = True
model_number = 1

db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
street_segment_capacity_limit = 10
prediction_window = 3
data_file_path = "./data/processed/classification_dataset_with_restrictions.csv"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
val_start_date = datetime.datetime(2019, 10, 1)
val_end_date = datetime.datetime(2019, 11, 1)
test_start_date = datetime.datetime(2019, 11, 1)
test_end_date = datetime.datetime(2020, 1, 1)
segments_sample_size = None

main_num_neurons = [1500, 1500, 1000]
main_epoch = 500
main_batch_size = 100
main_lr = 0.0001
main_loss = 'categorical_crossentropy'
main_activation = "tanh"
main_patience = 10
main_dropout_rate = 0.0
class_count = 5


if __name__ == '__main__':
    try:
        if not infer:
            model_number = 1
            model_dir_path = util_classification.make_model_dir(f"./models/RNN/{model_type.upper()}", "rnn_classification")
        else:
            model_dir_path = f"./models/RNN/{model_type.upper()}/model-" + str(model_number)

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
                                                   _3d_input=True,
                                                   _3d_output=False,
                                                   random_segments=segments_sample_size,
                                                   use_restrictions=True,
                                                   inject_noise=False,
                                                   noise_std=0.0,
                                                   noise_on_targets=False,
                                                   random_seed=42)

        if not infer:
            logger.debug(f"Creating model with specs:")
            logger.debug(f"model_type = {model_type}")
            logger.debug(f"main_num_neurons = {main_num_neurons}")
            logger.debug(f"main_epoch = {main_epoch}")
            logger.debug(f"main_batch_size = {main_batch_size}")
            logger.debug(f"main_lr = {main_lr}")
            logger.debug(f"main_loss = {main_loss}")
            logger.debug(f"main_activation = {main_activation}")
            logger.debug(f"main_patience = {main_patience}")
            logger.debug(f"prediction_window = {prediction_window}")
            logger.debug(f"class_count = {class_count}")

            predictor = util_classification.Rnn(model_type=model_type,
                                                train_x=dataset.train_X,
                                                train_y=dataset.train_y,
                                                valid_x=dataset.val_X,
                                                valid_y=dataset.val_y,
                                                num_neurons=main_num_neurons,
                                                epoch=main_epoch,
                                                batch_size=main_batch_size,
                                                lr=main_lr,
                                                loss=main_loss,
                                                activation=main_activation,
                                                patience=main_patience,
                                                class_count=class_count,
                                                prediction_window=prediction_window,
                                                dropout_rate=main_dropout_rate)

            logger.debug("Training main predictor")
            t = time.time()
            predictor.fit(log_file_path)
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

            predictor.save_model(model_dir_path)

        else:
            predictor = util_classification.Rnn(model_type=model_type,
                                                train_x=dataset.train_X,
                                                train_y=dataset.train_y,
                                                valid_x=dataset.val_X,
                                                valid_y=dataset.val_y,
                                                num_neurons=main_num_neurons,
                                                epoch=main_epoch,
                                                batch_size=main_batch_size,
                                                lr=main_lr,
                                                loss=main_loss,
                                                activation=main_activation,
                                                patience=main_patience,
                                                class_count=class_count,
                                                prediction_window=prediction_window,
                                                dropout_rate=main_dropout_rate)
            predictor.load_model(model_dir_path)

            main_y_hat = predictor.predict(dataset.test_X)

            dataset.test = util_classification.add_preds(dataset.test, main_y_hat, "RNN")

            util_classification.calculate_accuracy(dataset.test,
                                                   dataset.segment_ids,
                                                   prediction_window,
                                                   model_name="RNN",
                                                   model_path=model_dir_path)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)