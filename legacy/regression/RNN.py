import os
import time
import util
import logging
import datetime
import pandas as pd

model_type = "LSTM"

log_file_path = './logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower() + f'_{model_type.lower()}')
logging.basicConfig(filename=log_file_path,
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None
os.environ['NUMEXPR_MAX_THREADS'] = '30'

# Set predictor construction or inference mode
infer = False
model_number = 2

# Props.
db_name = "melbourne-parking"
year = 2019
agg_interval = 15
lookBack = 8
street_segment_capacity_limit = 10
prediction_window = 4
data_file_path = "./data/street_segments_occupancy_15min_2019.csv"
weather_data_path = "./data/historical_weather_data_2019-01-01_to_2020-01-01.csv"
train_start_date = datetime.datetime(2019, 1, 1)
train_end_date = datetime.datetime(2019, 10, 1)
val_start_date = datetime.datetime(2019, 10, 1)
val_end_date = datetime.datetime(2020, 1, 1)
test_start_date = datetime.datetime(2019, 10, 1)
test_end_date = datetime.datetime(2020, 1, 1)

# Main predictor props.
main_num_neurons = [1500, 1500]
main_epoch = 500
main_batch_size = 100
main_lr = 0.0001
main_loss = util.mse_loss_mean
main_activation = "tanh"
main_patience = 10


if __name__ == '__main__':
    try:
        # make model directory
        if not infer:
            model_number = 1
            model_dir_path = util.make_model_dir(f"./models/RNN/{model_type.upper()}", "RNN")
        else:
            model_dir_path = f"./models/RNN/{model_type.upper()}/model-" + str(model_number)

        # Load data
        dataset = util.GenerateLSTMDate(data_file_path=data_file_path, look_back=lookBack,
                                        train_start_date=train_start_date, train_end_date=train_end_date,
                                        val_start_date=val_start_date, val_end_date=val_end_date,
                                        test_start_date=test_start_date, test_end_date=test_end_date,
                                        prediction_window=prediction_window, weather_data_path=None,
                                        capacity_limit=street_segment_capacity_limit,
                                        interval=agg_interval, normalize=False)

        # Training Phase
        if not infer:
            # Build predictor

            logger.debug(f"Creating model with specs:\n"
                         f"model_type = {model_type}\n"
                         f"main_num_neurons = {main_num_neurons}\n"
                         f"main_epoch = {main_epoch}\n"
                         f"main_batch_size = {main_batch_size}\n"
                         f"main_lr = {main_lr}\n"
                         f"main_loss = {main_loss}\n"
                         f"main_activation = {main_activation}\n"
                         f"main_patience = {main_patience}")
            predictor = util.Rnn(train_x=dataset.train_X, train_y=dataset.train_y,
                                 valid_x=dataset.val_X, valid_y=dataset.val_y,
                                 num_neurons=main_num_neurons, epoch=main_epoch,
                                 batch_size=main_batch_size, lr=main_lr, loss=main_loss,
                                 activation=main_activation, patience=main_patience,
                                 num_segments=len(dataset.segment_ids), prediction_window=prediction_window)

            # Train predictor
            logger.debug("Training main predictor")
            t = time.time()
            predictor.fit(log_file_path)
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

            # Save trained predictor
            predictor.save_model(model_dir_path)

        # Inference Phase
        else:
            # Load predictor
            predictor = util.Rnn(model_type=model_type,
                                 train_x=dataset.train_X, train_y=dataset.train_y,
                                 valid_x=dataset.val_X, valid_y=dataset.val_y,
                                 num_neurons=main_num_neurons, epoch=main_epoch,
                                 batch_size=main_batch_size, lr=main_lr, loss=main_loss,
                                 activation=main_activation, patience=main_patience,
                                 num_segments=len(dataset.segment_ids))
            predictor.load_model(model_dir_path)

            # Predict
            main_y_hat = predictor.predict(dataset.test_X)

            # Add predictions to DataFrame
            # main_y_hat = main_y_hat.reshape((main_y_hat.shape[0]*main_y_hat.shape[1], main_y_hat.shape[2]))
            # main_y_hat = main_y_hat.reshape((main_y_hat.shape[0]*main_y_hat.shape[1], main_y_hat.shape[2]))
            dataset.test = util.add_preds(dataset.test, main_y_hat, "RNN", dataset.segment_ids)

            # # De-normalize data
            # dataset.test = util.denormalize_data(dataset.test, ['region', 'RNN'], dataset.mean, dataset.std)

            # # Round predictions
            # dataset.test = util.round_values(dataset.test, ['RNN'])

            # Calculate Errors
            util.calculate_errors(dataset.test, pred="RNN")

            # Write results out
            util.result_file(dataset.test, model_path=model_dir_path, model_names=["RNN"])

            # Write predicted data out
            util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="RNN")
    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)
