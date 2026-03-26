import os
import logging
import numpy as np
import pandas as pd
from math import sqrt
from shutil import copyfile
import sklearn.metrics as skmets

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, TimeDistributed, SimpleRNN, GRU, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import optimizers
from tensorflow.keras import losses

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


def normalize(x, mean, sd):
    return (x - mean) / sd


def denormalize(x, mean, sd):
    return x * sd + mean


def mape(y_true, y_pred):
    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df = df[df['y_true'] != 0]
    y_true = np.array(df['y_true'])
    y_pred = np.array(df['y_pred'])
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_errors(data, pred):
    target_labels = [column for column in data.columns if (column.startswith('segment') and ('-(t+' in column))]
    pred_labels = [col for col in data if col.startswith(pred)]
    targets = np.array(data[target_labels])
    preds = np.array(data[pred_labels])
    logger.debug(f"targets shape: {targets.shape}")
    logger.debug(f"preds shape: {preds.shape}")
    targets = targets.flatten()
    preds = preds.flatten()
    logger.debug(f"Errors for {pred} :\n")
    logger.debug(f"MAE: {skmets.mean_absolute_error(targets, preds)} \n")
    logger.debug(f"RMSE: {sqrt(skmets.mean_squared_error(targets, preds))} \n")
    logger.debug(f"MAPE: {mape(targets, preds)} \n")


def calculate_errors_raw(targets, preds, model_name):
    logger.debug(f"targets shape: {targets.shape}")
    logger.debug(f"preds shape: {preds.shape}")
    targets = targets.flatten()
    preds = preds.flatten()
    logger.debug(f"Errors for {model_name} :\n")
    logger.debug(f"MAE: {skmets.mean_absolute_error(targets, preds)} \n")
    logger.debug(f"RMSE: {sqrt(skmets.mean_squared_error(targets, preds))} \n")
    logger.debug(f"MAPE: {mape(targets, preds)} \n")


def calculate_arrays_errors(targets, preds):
    mae = skmets.mean_absolute_error(targets, preds)
    rmse = sqrt(skmets.mean_squared_error(targets, preds))
    MAPE = mape(targets, preds)
    logger.debug("MAE : {}   RMSE: {}    MAPE: {}".format(mae, rmse, MAPE))


def mse_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_loss = K.square(error)
    return squared_loss


def mse_loss_mean(y_true, y_pred):
    return K.mean(mse_loss(y_true, y_pred))


def result_file(data, model_path, model_names):
    target_labels = [column for column in data.columns if (column.startswith('segment') and ('-(t+' in column))]
    output_text = model_path + "/results.txt"
    f = open(output_text, 'w')
    for model_name in model_names:
        pred_labels = [col for col in data if col.startswith(model_name)]
        targets = np.array(data[target_labels]).flatten()
        preds = np.array(data[pred_labels]).flatten()
        f.write("{}:\n".format(model_name))
        f.write("MAE : {}\n".format(skmets.mean_absolute_error(targets, preds)))
        f.write("RMSE: {}\n".format(sqrt(skmets.mean_squared_error(targets, preds))))
        f.write("MAPE: {}\n".format(mape(targets, preds)))
    f.close()


def result_file_raw(targets, preds, model_path, model_names):
    output_text = model_path + "/results.txt"
    f = open(output_text, 'w')
    targets = targets.flatten()
    for i in range(len(model_names)):
        model_name = model_names[i]
        model_preds = preds[i].flatten()
        f.write("{}:\n".format(model_name))
        f.write("MAE : {}\n".format(skmets.mean_absolute_error(targets, model_preds)))
        f.write("RMSE: {}\n".format(sqrt(skmets.mean_squared_error(targets, model_preds))))
        f.write("MAPE: {}\n".format(mape(targets, model_preds)))
    f.close()


def make_model_dir(model_dir_path, code_file_name):
    directory_created = False
    model_number = 1
    dir_path = model_dir_path
    while not directory_created:
        model_dir_path = dir_path + "/model-" + str(int(model_number))
        if os.path.exists(model_dir_path):
            model_number = model_number + 1
        else:
            os.makedirs(model_dir_path)
            directory_created = True
    copyfile("./" + code_file_name + ".py", model_dir_path + "/" + code_file_name + ".py")
    return model_dir_path


class GenerateLSTMDate:
    def __init__(self, data_file_path, look_back, interval,
                 train_start_date, train_end_date,
                 test_start_date, test_end_date,
                 prediction_window=1, weather_data_path=None,
                 capacity_limit=0, normalize=False,
                 val_start_date=None, val_end_date=None):
        self.data_file_path = data_file_path
        self.interval = interval
        self.lookBack = look_back
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.prediction_window = prediction_window
        self.weather_data_path = weather_data_path
        self.capacity_limit = capacity_limit
        self.temporal_columns = []
        self.weather_columns = []
        self.segment_current_columns = []
        self.segment_ids = None
        self.dataset = None
        self.train = None
        self.val = None
        self.test = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.test_y = None
        self.y_columns = None
        self.x_columns = None
        self.load_dataset()
        if self.weather_data_path is not None:
            self.add_weather_data()
        # if normalize:
        #     self.normalize_data()
        self.create_sequential_data()
        self.split_Train_Test()
        self.split_input_output_reshape()

    def load_dataset(self):
        dataset = pd.read_csv(self.data_file_path)
        dataset = dataset[dataset.capacity > self.capacity_limit]
        logger.debug('csv file loaded')
        dataset.timeslot = pd.to_datetime(dataset.timeslot)
        dataset['segment_id'] = dataset['streetid'].astype('str') + \
                                '-' + dataset['betweenstreet1id'].astype('str') + \
                                '-' + dataset['betweenstreet2id'].astype('str')
        logger.debug('creating pivot table')
        pivot_dataset = pd.pivot_table(dataset, values='occupancy_ratio',
                                       index='timeslot',
                                       columns=['segment_id'])
        logger.debug('pivot table created')
        pivot_dataset.reset_index(inplace=True)
        pivot_dataset.columns = list(map(lambda x: x.replace('timeslot', 'date_time'), pivot_dataset.columns))

        pivot_dataset['day_of_week_sin-(t)'] \
            = pivot_dataset.date_time.apply(lambda x: np.sin(x.dayofweek * (2 * np.pi / 7)))
        pivot_dataset['day_of_week_cos-(t)'] \
            = pivot_dataset.date_time.apply(lambda x: np.cos(x.dayofweek * (2 * np.pi / 7)))

        pivot_dataset['timeslot-(t)'] \
            = pivot_dataset.date_time.apply(lambda x: int(x.minute / self.interval + x.hour * 60 / self.interval))
        pivot_dataset['timeslot_sin-(t)'] = \
            pivot_dataset['timeslot-(t)'].apply(lambda x: np.sin(x * (2 * np.pi / (24 * 60 / self.interval))))
        pivot_dataset['timeslot_cos-(t)'] = \
            pivot_dataset['timeslot-(t)'].apply(lambda x: np.cos(x * (2 * np.pi / (24 * 60 / self.interval))))
        pivot_dataset.drop(['timeslot-(t)'], axis=1, inplace=True)
        pivot_dataset = pivot_dataset.reindex(sorted(pivot_dataset.columns, reverse=True), axis=1)

        self.temporal_columns = ['day_of_week_sin-(t)', 'day_of_week_cos-(t)',
                                 'timeslot_sin-(t)', 'timeslot_cos-(t)']

        self.segment_ids \
            = [x for x in pivot_dataset.columns if x not in self.temporal_columns + ['date_time']]
        new_columns = [f'segment_{column}-(t)' if (column not in ['date_time'] + self.temporal_columns)
                       else column for column in pivot_dataset.columns]
        pivot_dataset.columns = new_columns

        self.segment_current_columns = [column for column in pivot_dataset.columns
                                        if (column not in ['date_time'] + self.temporal_columns)]
        pivot_dataset.dropna(inplace=True)
        self.dataset = pivot_dataset
        logger.debug('dataset preprocessed')

    def add_weather_data(self):
        logger.debug("Processing weather data")
        weather_data = pd.read_csv(self.weather_data_path, usecols=["DateTime", "Temperature", "Humidity",
                                                                    "Wind Speed", "Condition"])
        weather_data.DateTime = pd.to_datetime(weather_data.DateTime)
        weather_data.columns = [col.replace(' ', '_') for col in weather_data.columns]
        weather_data.Temperature.fillna(method='ffill', inplace=True)
        weather_data.Humidity.fillna(method='ffill', inplace=True)
        weather_data.Wind_Speed.fillna(method='ffill', inplace=True)
        weather_data.Condition.fillna(method='ffill', inplace=True)
        sample_record = weather_data.iloc[0]

        if "F" in sample_record.Temperature:
            weather_data.Temperature = weather_data.Temperature.apply(lambda x: int(x.split('\xa0')[0]))
        max_temp = weather_data.Temperature.max()
        min_temp = weather_data.Temperature.min()
        weather_data.Temperature = (weather_data.Temperature - min_temp) / (max_temp - min_temp)

        weather_data.Humidity = weather_data.Humidity.apply(lambda x: int(x.split('\xa0')[0]) / 100)

        if "mph" in sample_record.Wind_Speed:
            weather_data.Wind_Speed = weather_data.Wind_Speed.apply(lambda x: int(x.split('\xa0')[0]))
        max_wind = weather_data.Wind_Speed.max()
        min_wind = weather_data.Wind_Speed.min()
        weather_data.Wind_Speed = (weather_data.Wind_Speed - min_wind) / (max_wind - min_wind)

        weather_data.Condition = weather_data.Condition.apply(lambda x: x.replace(' / ', '/'))
        weather_data.Condition = weather_data.Condition.apply(lambda x: x.replace(' ', ''))
        weather_data = pd.concat([weather_data, pd.get_dummies(weather_data.Condition, prefix="Condition")], axis=1)
        weather_data.drop(['Condition'], axis=1, inplace=True)

        weather_data.set_index(['DateTime'], inplace=True)
        weather_data.columns = ['weather_' + col + '-(t)' for col in weather_data.columns]
        self.weather_columns = list(weather_data.columns)

        weather_data.dropna(inplace=True)
        weather_data.reset_index(inplace=True)
        weather_data.columns = ['date_time' if (col == 'DateTime') else col for col in weather_data.columns]
        logger.debug(f"weather data processed / shape: {weather_data.shape}")
        self.dataset = self.dataset.merge(weather_data, on='date_time')
        logger.debug(f'weather data added to dataset / shape: {self.dataset.shape}')

    def create_sequential_data(self):
        self.dataset.set_index(['date_time'], inplace=True)
        x_columns = self.temporal_columns + self.weather_columns + self.segment_current_columns
        for i in range(self.lookBack - 1, 0, -1):
            for x_column in x_columns:
                self.dataset[x_column.replace('(t)', f'(t-{i})')] = self.dataset[x_column].shift(i)

        for i in range(0, self.prediction_window):
            for segment_id in self.segment_ids:
                self.dataset[f'segment_{segment_id}-(t+{i + 1})'] \
                    = self.dataset[f'segment_{segment_id}-(t)'].shift(-(i + 1))
        self.dataset.dropna(inplace=True)
        self.dataset.reset_index(inplace=True)

    # def normalize_data(self):
    #     self.dataset.dropna(axis=0, how='any', inplace=True)
    #     targets_label = [col for col in self.dataset.columns if col.startswith('segment')]
    #     labels = list(targets_label)
    #     for label in labels:
    #         self.dataset[label] = self.dataset[label].apply(lambda x: normalize(x, self.mean, self.std))

    def split_Train_Test(self):
        logger.debug('splitting dataset')
        self.test = self.dataset[(self.dataset['date_time'] >= self.test_start_date) &
                                 (self.dataset['date_time'] < self.test_end_date)]
        self.train = self.dataset[(self.dataset['date_time'] >= self.train_start_date) &
                                  (self.dataset['date_time'] < self.train_end_date)]
        if self.val_start_date is not None and self.val_end_date is not None:
            self.val = self.dataset[(self.dataset['date_time'] >= self.val_start_date) &
                                    (self.dataset['date_time'] < self.val_end_date)]
        logger.debug('dataset split')

    def split_input_output_reshape(self):
        logger.debug('reshaping data')
        self.y_columns = [column for column in self.dataset.columns if ('-(t+' in column)]
        self.x_columns = [column for column in self.dataset.columns if (('-(t-' in column) or ('-(t)' in column))]

        # seq_len = self.lookBack
        #
        # self.train = self.train[self.train.shape[0] % seq_len:]
        # self.test = self.test[self.test.shape[0] % seq_len:]
        # if self.val_start_date is not None and self.val_end_date is not None:
        #     self.val = self.val[self.val.shape[0] % seq_len:]

        train_X, train_y = self.train[self.x_columns].values, self.train[self.y_columns].values
        test_X, test_y = self.test[self.x_columns].values, self.test[self.y_columns].values
        if self.val_start_date is not None and self.val_end_date is not None:
            val_X, val_y = self.val[self.x_columns].values, self.val[self.y_columns].values

        self.train_X = train_X.reshape((train_X.shape[0], self.lookBack, int(train_X.shape[1] / self.lookBack)))
        self.train_y = train_y
        # self.train_y = train_y.reshape((train_y.shape[0],
        #                                 self.prediction_window,
        #                                 int(train_y.shape[1] / self.prediction_window)))
        self.test_X = test_X.reshape((test_X.shape[0], self.lookBack, int(test_X.shape[1] / self.lookBack)))
        self.test_y = test_y
        # self.test_y = test_y.reshape((test_y.shape[0],
        #                               self.prediction_window,
        #                               int(test_y.shape[1] / self.prediction_window)))
        if self.val_start_date is not None and self.val_end_date is not None:
            self.val_X = val_X.reshape((val_X.shape[0], self.lookBack, int(val_X.shape[1] / self.lookBack)))
            self.val_y = val_y
            # self.val_y = val_y.reshape((val_y.shape[0],
            #                             self.prediction_window,
            #                             int(val_y.shape[1] / self.prediction_window)))
        # self.train_X = train_X.reshape((int(train_X.shape[0] / seq_len), seq_len, train_X.shape[1]))
        # self.train_y = train_y.reshape((int(train_y.shape[0] / seq_len), seq_len, train_y.shape[1]))
        # self.test_X = test_X.reshape((int(test_X.shape[0] / seq_len), seq_len, test_X.shape[1]))
        # self.test_y = test_y.reshape((int(test_y.shape[0] / seq_len), seq_len, test_y.shape[1]))
        # if self.val_start_date is not None and self.val_end_date is not None:
        #     self.val_X = val_X.reshape((int(val_X.shape[0] / seq_len), seq_len, val_X.shape[1]))
        #     self.val_y = val_y.reshape((int(val_y.shape[0] / seq_len), seq_len, val_y.shape[1]))
        logger.debug('data reshaped')


class GenerateOtherMethodsData:
    def __init__(self, data_file_path, look_back, interval,
                 train_start_date, train_end_date,
                 test_start_date, test_end_date,
                 prediction_window=1, weather_data_path=None,
                 capacity_limit=0, input_3d=True,
                 val_start_date=None, val_end_date=None):
        # self.db_engine = db_engine
        # self.table_name = table_name
        self.data_file_path = data_file_path
        self.lookBack = look_back
        self.interval = interval
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.prediction_window = prediction_window
        self.weather_data_path = weather_data_path
        self.capacity_limit = capacity_limit
        self.input_3d = input_3d
        self.segment_ids = None
        self.dataset = None
        self.train = None
        self.val = None
        self.test = None
        self.train_X = None
        self.train_y = None
        self.val_X = None
        self.val_y = None
        self.test_X = None
        self.test_y = None
        self.y_columns = None
        self.x_columns = None
        self.load_dataset()
        if self.weather_data_path is not None:
            self.add_weather_data()
        self.split_train_test()
        self.split_input_output_reshape()

    def load_dataset(self):
        # cmd = f"""select *
        #             from {self.table_name}
        #             where capacity > 10"""
        # dataset = pd.read_sql(cmd, self.db_engine)
        dataset = pd.read_csv(self.data_file_path)
        dataset = dataset[dataset.capacity > self.capacity_limit]
        logger.debug('csv file loaded')
        dataset.timeslot = pd.to_datetime(dataset.timeslot)
        dataset['segment_id'] = dataset['streetid'].astype('str') +\
                                '-' + dataset['betweenstreet1id'].astype('str') +\
                                '-' + dataset['betweenstreet2id'].astype('str')
        logger.debug('creating pivot table')
        pivot_dataset = pd.pivot_table(dataset, values='occupancy_ratio',
                                       index='timeslot',
                                       columns=['segment_id'])
        logger.debug('pivot table created')
        pivot_dataset.reset_index(inplace=True)
        pivot_dataset.columns = list(map(lambda x: x.replace('timeslot', 'date_time'), pivot_dataset.columns))
        pivot_dataset['day_of_week'] = pivot_dataset.date_time.apply(lambda x: x.dayofweek / 6)
        pivot_dataset['timeslot'] \
            = pivot_dataset.date_time.apply(lambda x: int(x.minute / self.interval + x.hour * 60 / self.interval))
        max_timeslot = 24 * 60 / self.interval
        pivot_dataset['timeslot'] = pivot_dataset.timeslot / max_timeslot
        pivot_dataset = pivot_dataset.reindex(sorted(pivot_dataset.columns, reverse=True), axis=1)
        self.segment_ids = [x for x in pivot_dataset.columns if x not in ['timeslot', 'day_of_week', 'date_time']]
        new_columns = [f'segment_{column}-(t)' if (column not in ['timeslot', 'day_of_week', 'date_time'])
                       else column for column in pivot_dataset.columns]
        pivot_dataset.columns = new_columns

        for i in range(self.lookBack - 1, 0, -1):
            for segment_id in self.segment_ids:
                pivot_dataset[f'segment_{segment_id}-(t-{i})'] = pivot_dataset[f'segment_{segment_id}-(t)'].shift(i)

        for i in range(0, self.prediction_window):
            for segment_id in self.segment_ids:
                pivot_dataset[f'segment_{segment_id}-(t+{i+1})'] = \
                    pivot_dataset[f'segment_{segment_id}-(t)'].shift(-(i+1))

        pivot_dataset.dropna(inplace=True)
        self.dataset = pivot_dataset
        logger.debug(f'dataset preprocessed / shape: {self.dataset.shape}')

    def add_weather_data(self):
        logger.debug("Processing weather data")
        weather_data = pd.read_csv(self.weather_data_path, usecols=["DateTime", "Temperature", "Humidity",
                                                                    "Wind Speed", "Condition"])
        weather_data.DateTime = pd.to_datetime(weather_data.DateTime)
        weather_data.columns = [col.replace(' ', '_') for col in weather_data.columns]
        weather_data.Temperature.fillna(method='ffill', inplace=True)
        weather_data.Humidity.fillna(method='ffill', inplace=True)
        weather_data.Wind_Speed.fillna(method='ffill', inplace=True)
        weather_data.Condition.fillna(method='ffill', inplace=True)
        sample_record = weather_data.iloc[0]

        if "F" in sample_record.Temperature:
            weather_data.Temperature = weather_data.Temperature.apply(lambda x: int(x.split('\xa0')[0]))
        max_temp = weather_data.Temperature.max()
        min_temp = weather_data.Temperature.min()
        weather_data.Temperature = (weather_data.Temperature - min_temp) / (max_temp - min_temp)

        weather_data.Humidity = weather_data.Humidity.apply(lambda x: int(x.split('\xa0')[0]) / 100)

        if "mph" in sample_record.Wind_Speed:
            weather_data.Wind_Speed = weather_data.Wind_Speed.apply(lambda x: int(x.split('\xa0')[0]))
        max_wind = weather_data.Wind_Speed.max()
        min_wind = weather_data.Wind_Speed.min()
        weather_data.Wind_Speed = (weather_data.Wind_Speed - min_wind) / (max_wind - min_wind)

        weather_data.Condition = weather_data.Condition.apply(lambda x: x.replace(' / ', '/'))
        weather_data.Condition = weather_data.Condition.apply(lambda x: x.replace(' ', ''))
        weather_data = pd.concat([weather_data, pd.get_dummies(weather_data.Condition, prefix="Condition")], axis=1)
        weather_data.drop(['Condition'], axis=1, inplace=True)

        weather_data.set_index(['DateTime'], inplace=True)
        weather_main_cols = weather_data.columns
        weather_data.columns = ['weather_' + col + '-(t)' for col in weather_data.columns]

        for i in range(self.lookBack - 1, 0, -1):
            for col in weather_main_cols:
                weather_data['weather_' + col + f'-(t-{i})'] = weather_data[f'weather_{col}-(t)'].shift(i)

        weather_data.dropna(inplace=True)
        weather_data.reset_index(inplace=True)
        weather_data.columns = ['date_time' if (col == 'DateTime') else col for col in weather_data.columns]
        logger.debug(f"weather data processed / shape: {weather_data.shape}")
        self.dataset = self.dataset.merge(weather_data, on='date_time')
        logger.debug(f'weather data added to dataset / shape: {self.dataset.shape}')

    def split_train_test(self):
        self.train = self.dataset[(self.dataset['date_time'] >= self.train_start_date) &
                                  (self.dataset['date_time'] < self.train_end_date)]
        if self.val_start_date and self.val_end_date:
            self.val = self.dataset[(self.dataset['date_time'] >= self.val_start_date) &
                                    (self.dataset['date_time'] < self.val_end_date)]
        self.test = self.dataset[(self.dataset['date_time'] >= self.test_start_date) &
                                 (self.dataset['date_time'] < self.test_end_date)]

    def split_input_output_reshape(self):
        self.y_columns = [column for column in self.dataset.columns if ('-(t+' in column)]
        self.x_columns = ['timeslot', 'day_of_week'] + \
                         [column for column in self.dataset.columns if (('-(t-' in column) or ('-(t)' in column))]
        # logger.debug(f"Y columns: {len(self.y_columns)}")
        # logger.debug("\n".join(self.y_columns))
        # logger.debug(f"X columns: {len(self.x_columns)}")
        # logger.debug("\n".join(self.x_columns))
        train_X, train_y = self.train[self.x_columns].values, self.train[self.y_columns].values
        if self.val_start_date and self.val_end_date:
            val_X, val_y = self.val[self.x_columns].values, self.val[self.y_columns].values
        test_X, test_y = self.test[self.x_columns].values, self.test[self.y_columns].values
        if self.input_3d:
            self.train_X = train_X.reshape((train_X.shape[0], self.lookBack, int(train_X.shape[1] / self.lookBack)))
            if self.val_start_date and self.val_end_date:
                self.val_X = val_X.reshape((val_X.shape[0], self.lookBack, int(val_X.shape[1] / self.lookBack)))
            self.test_X = test_X.reshape((test_X.shape[0], self.lookBack, int(test_X.shape[1] / self.lookBack)))
        else:
            self.train_X = train_X
            if self.val_start_date and self.val_end_date:
                self.val_X = val_X
            self.test_X = test_X
        self.train_y = train_y.reshape((train_y.shape[0], len(self.segment_ids)))
        if self.val_start_date and self.val_end_date:
            self.val_y = val_y.reshape((val_y.shape[0], len(self.segment_ids)))
        self.test_y = test_y.reshape((test_y.shape[0], len(self.segment_ids)))


def add_preds(data, preds, label, segments, prediction_window=1):
    pred_cols = []
    for prediction_step in range(prediction_window):
        for segment_id in segments:
            col_name = label + "_" + str(segment_id) + "-" + f"(t+{prediction_step + 1})"
            data[col_name] = None
            pred_cols.append(col_name)
        # for i in range(len(segments), 0, -1):
        #     data[data.columns[-i]] = preds[:, len(segments) - i]
    data.loc[:, pred_cols] = preds
    return data


def round_values(data, labels):
    all_labels = []
    for label in labels:
        all_labels += [col for col in data.columns if col.startswith(label)]
    for label in all_labels:
        data[label] = data[label].apply(lambda x: round(x))
    return data


def predictions_to_csv(data, model_dir_path, file_name):
    data.to_csv(model_dir_path + "/" + file_name + "_preds.csv", index=False)


def predictions_to_csv_raw(preds, model_dir_path, model_name):
    np.savetxt(model_dir_path + "/" + model_name + "_preds.csv", preds, delimiter=",")


class Rnn:
    def __init__(self, model_type="simple", train_x=None, train_y=None, valid_x=None, valid_y=None, num_neurons=None,
                 epoch=100, batch_size=20, lr=0.01, loss=losses.mean_squared_error,
                 activation='tanh', patience=10, num_segments=None, prediction_window=1):
        # Model properties
        if model_type.lower() not in ["simple", "gru", "lstm"]:
            raise Exception("RNN should be specified and should be 'SIMPLE', 'GRU' OR 'LSTM'")
        if train_x is None:
            raise Exception("Please specify train_x data")
        if train_y is None:
            raise Exception("Please specify train_y data")
        if valid_x is None:
            raise Exception("Please specify valid_x data")
        if valid_y is None:
            raise Exception("Please specify valid_y data")
        if num_neurons is None:
            raise Exception("Please specify num_neurons")
        if num_segments is None:
            raise Exception("Please specify num_segments")

        self.model_type = model_type.lower()
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_segments = num_segments
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()

        if self.model_type == 'simple':
            model.add(SimpleRNN(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                                return_sequences=True, activation=self.act_func))
        elif self.model_type == 'gru':
            model.add(GRU(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                          return_sequences=True, activation=self.act_func))
        elif self.model_type == 'lstm':
            model.add(LSTM(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
        else:
            raise Exception("model_type should be 'SIMPLE', 'GRU' OR 'LSTM")
        if len(self.num_neurons) > 1:
            for layer in self.num_neurons[1:len(self.num_neurons) - 1]:
                if self.model_type == 'simple':
                    model.add(SimpleRNN(layer, return_sequences=True, activation=self.act_func))
                elif self.model_type == 'gru':
                    model.add(GRU(layer, return_sequences=True, activation=self.act_func))
                elif self.model_type == 'lstm':
                    model.add(LSTM(layer, return_sequences=True, activation=self.act_func))
            if self.model_type == 'simple':
                model.add(SimpleRNN(self.num_neurons[-1], return_sequences=False, activation=self.act_func))
            elif self.model_type == 'gru':
                model.add(GRU(self.num_neurons[-1], return_sequences=False, activation=self.act_func))
            elif self.model_type == 'lstm':
                model.add(LSTM(self.num_neurons[-1], return_sequences=False, activation=self.act_func))

        model.add(Dense(self.num_segments * self.prediction_window, activation='sigmoid'))
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self, log_file_path):
        csv_logger = CSVLogger(log_file_path, append=True, separator=';')
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                     csv_logger]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", segment_number=""):
        if (model_id == "") and (segment_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif segment_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "segment_" + str(segment_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")

        hist_df = pd.DataFrame(self.history.history)
        with open(self.predictor_file_path + '_history.csv', 'wb') as file_pi:
            hist_df.to_csv(file_pi)
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", segment_number=""):
        if (model_id == "") and (segment_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            logger.debug("Loading main predictor")
        elif segment_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            logger.debug("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "segment_" + str(segment_number)
            logger.debug("Loading predictor number {} for segment number {}".format(model_id, segment_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


class Ann:
    def __init__(self, model_type="ANN", train_x=None, train_y=None, valid_x=None, valid_y=None, num_neurons=None,
                 epoch=100, batch_size=20, lr=0.01, loss=losses.mean_squared_error,
                 activation='tanh', patience=10, num_segments=None, prediction_window=1):
        # Model properties
        if model_type.lower() not in ["simple", "gru", "lstm", "ann"]:
            raise Exception("RNN should be specified and should be 'SIMPLE', 'GRU' OR 'LSTM'")
        if train_x is None:
            raise Exception("Please specify train_x data")
        if train_y is None:
            raise Exception("Please specify train_y data")
        if valid_x is None:
            raise Exception("Please specify valid_x data")
        if valid_y is None:
            raise Exception("Please specify valid_y data")
        if num_neurons is None:
            raise Exception("Please specify num_neurons")
        if num_segments is None:
            raise Exception("Please specify num_segments")

        self.model_type = model_type.lower()
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_segments = num_segments
        self.prediction_window = prediction_window
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        for layer in self.num_neurons:
            model.add(Dense(layer, activation=self.act_func))
        model.add(Dense(self.num_segments * self.prediction_window, activation='sigmoid'))
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self, log_file_path):
        csv_logger = CSVLogger(log_file_path, append=True, separator=';')
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                     csv_logger]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", segment_number=""):
        if (model_id == "") and (segment_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif segment_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "segment_" + str(segment_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")

        hist_df = pd.DataFrame(self.history.history)
        with open(self.predictor_file_path + '_history.csv', 'wb') as file_pi:
            hist_df.to_csv(file_pi)
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", segment_number=""):
        if (model_id == "") and (segment_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            logger.debug("Loading main predictor")
        elif segment_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            logger.debug("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "segment_" + str(segment_number)
            logger.debug("Loading predictor number {} for segment number {}".format(model_id, segment_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model
