import os
import logging
import numpy as np
import pandas as pd
from shutil import copyfile
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


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
    try:
        copyfile("./" + code_file_name + ".py", model_dir_path + "/" + code_file_name + ".py")
    except BaseException:
        pass
    return model_dir_path


def add_preds(data, preds, label):
    pred_cols = []
    if len(preds.shape) == 1:
        preds = preds.reshape((-1, 1))
    for i in range(preds.shape[1]):
        col_name = label + f"_target_{i + 1}"
        data[col_name] = preds[:, i]
        pred_cols.append(col_name)
    return data


def add_prob_preds(data, preds, label):
    if len(preds.shape) == 3:
        for i in range(preds.shape[1]):
            for j in range(preds.shape[2]):
                col_name = label + f"_target_{i + 1}_class_{j}"
                data[col_name] = preds[:, i, j]
    return data


def accuracy_at_one_score(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred) <= 1)


def calculate_accuracy(data, segment_ids=None, prediction_window=1, model_name=None, model_path=None):
    target_cols = [x for x in data.columns if x.startswith("target_class_t+")]
    pred_cols = [x for x in data.columns if x.startswith(model_name + "_target_")]

    if len(target_cols) == 0 or len(pred_cols) == 0:
        return

    target_cols = sorted(target_cols, key=lambda x: int(x.split("+")[1].split("]")[0]) if "]" in x else int(x.split("+")[1]))
    pred_cols = sorted(pred_cols, key=lambda x: int(x.split("_")[-1]))

    output_lines = []

    for i in range(min(len(target_cols), len(pred_cols))):
        y_true = data[target_cols[i]].dropna().astype(int)
        y_pred = data.loc[y_true.index, pred_cols[i]].astype(int)

        acc = accuracy_score(y_true, y_pred)
        acc1 = accuracy_at_one_score(y_true, y_pred)
        pred_ratio = y_pred.notna().mean()

        output_lines.append(f"Prediction Window {i + 1}")
        output_lines.append(f"Prediction Ratio: {pred_ratio}")
        output_lines.append(f"Accuracy: {acc}")
        output_lines.append(f"Accuracy@1: {acc1}")
        output_lines.append("")

        logger.debug(f"{model_name} - Prediction Window {i + 1}")
        logger.debug(f"Prediction Ratio: {pred_ratio}")
        logger.debug(f"Accuracy: {acc}")
        logger.debug(f"Accuracy@1: {acc1}")

    if model_path is not None:
        with open(model_path + "/results.txt", "w") as f:
            f.write("\n".join(output_lines))


class GenerateDate:
    def __init__(self, data_file_path=None, look_back=8,
                 train_start_date=None, train_end_date=None,
                 test_start_date=None, test_end_date=None,
                 prediction_window=3, weather_data_path=None,
                 capacity_limit=0, interval=15, normalize_flag=False,
                 classification=True, one_hot_encoding=False,
                 _3d_input=True, _3d_output=False,
                 random_segments=None, use_restrictions=True,
                 val_start_date=None, val_end_date=None,
                 inject_noise=False, noise_std=0.0,
                 noise_on_targets=False, random_seed=42):
        self.data_file_path = data_file_path
        self.look_back = look_back
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.prediction_window = prediction_window
        self.weather_data_path = weather_data_path
        self.capacity_limit = capacity_limit
        self.interval = interval
        self.normalize_flag = normalize_flag
        self.classification = classification
        self.one_hot_encoding = one_hot_encoding
        self._3d_input = _3d_input
        self._3d_output = _3d_output
        self.random_segments = random_segments
        self.use_restrictions = use_restrictions
        self.inject_noise = inject_noise
        self.noise_std = noise_std
        self.noise_on_targets = noise_on_targets
        self.random_seed = random_seed

        self.dataset = None
        self.segment_ids = None
        self.feature_columns = None
        self.target_columns = None

        self.train = None
        self.test = None
        self.val = None

        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.val_X = None
        self.val_y = None

        self.load_dataset()
        self.split_train_test()
        self.split_input_output_reshape()

    def load_dataset(self):
        df = pd.read_csv(self.data_file_path)
        if "timeslot" in df.columns:
            df["timeslot"] = pd.to_datetime(df["timeslot"])

        if "segment_id" not in df.columns and all(x in df.columns for x in ["streetid", "betweenstreet1id", "betweenstreet2id"]):
            df["segment_id"] = df["streetid"].astype(str) + "-" + df["betweenstreet1id"].astype(str) + "-" + df["betweenstreet2id"].astype(str)

        if self.random_segments is not None and "segment_id" in df.columns:
            unique_segments = sorted(df["segment_id"].dropna().unique().tolist())
            if self.random_segments < len(unique_segments):
                np.random.seed(42)
                selected = np.random.choice(unique_segments, self.random_segments, replace=False)
                df = df[df["segment_id"].isin(selected)]

        if not self.use_restrictions:
            restriction_cols = [x for x in df.columns if (
                "restriction" in x.lower() or
                "typedesc" in x.lower() or
                "description" in x.lower() or
                "duration_" in x.lower() or
                "effectiveonph" in x.lower() or
                "meter_ratio" in x.lower() or
                "disabled_ratio" in x.lower() or
                "loading_ratio" in x.lower() or
                "permit_ratio" in x.lower() or
                "clearway_ratio" in x.lower() or
                "no_parking_ratio" in x.lower() or
                "no_stopping_ratio" in x.lower() or
                "active_restriction_ratio" in x.lower() or
                "restricted_bay_count" in x.lower() or
                "disability_ext_" in x.lower()
            )]
            df.drop(columns=[x for x in restriction_cols if x in df.columns], inplace=True)

        df.sort_values(["segment_id", "timeslot"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.segment_ids = sorted(df["segment_id"].dropna().unique().tolist()) if "segment_id" in df.columns else None
        self.dataset = df

    def split_train_test(self):
        self.train = self.dataset[(self.dataset["timeslot"] >= self.train_start_date) &
                                  (self.dataset["timeslot"] < self.train_end_date)]
        self.test = self.dataset[(self.dataset["timeslot"] >= self.test_start_date) &
                                 (self.dataset["timeslot"] < self.test_end_date)]
        if self.val_start_date is not None and self.val_end_date is not None:
            self.val = self.dataset[(self.dataset["timeslot"] >= self.val_start_date) &
                                    (self.dataset["timeslot"] < self.val_end_date)]

    def get_columns(self):
        target_columns = [x for x in self.dataset.columns if x.startswith("target_class_t+")]
        ignore_cols = ["segment_id", "timeslot", "streetid", "betweenstreet1id", "betweenstreet2id"]
        feature_columns = [x for x in self.dataset.columns if x not in ignore_cols + target_columns and not x.startswith("target_ratio_t+")]

        feature_columns = sorted(feature_columns)
        target_columns = sorted(target_columns, key=lambda x: int(x.split("+")[1]))

        return feature_columns, target_columns

    def reshape_input(self, X):
        if not self._3d_input:
            return X

        occupancy_cols = [x for x in self.feature_columns if x.startswith("occupancy_ratio_t-") or x.startswith("occupancy_class_t-")]
        static_cols = [x for x in self.feature_columns if x not in occupancy_cols]

        if len(occupancy_cols) == 0:
            return X

        occupancy_cols = sorted(occupancy_cols, key=lambda x: int(x.split("-")[1]))
        static_part = X[:, [self.feature_columns.index(x) for x in static_cols]] if len(static_cols) > 0 else None

        time_steps = []
        for step in range(self.look_back, 0, -1):
            step_cols = [x for x in occupancy_cols if x.endswith(f"t-{step}")]
            step_cols = sorted(step_cols)
            step_data = X[:, [self.feature_columns.index(x) for x in step_cols]]
            if static_part is not None:
                step_data = np.concatenate([step_data, static_part], axis=1)
            time_steps.append(step_data)

        X_3d = np.stack(time_steps, axis=1)
        return X_3d

    def reshape_output(self, y):
        if self.one_hot_encoding:
            class_count = len(np.unique(self.dataset[self.target_columns].values.flatten()))
            y_new = []
            for i in range(y.shape[1]):
                y_new.append(to_categorical(y[:, i], num_classes=class_count))
            y = np.stack(y_new, axis=1)

        if self._3d_output and not self.one_hot_encoding:
            y = y.reshape((y.shape[0], self.prediction_window, int(y.shape[1] / self.prediction_window)))

        return y
      
    def apply_gaussian_noise(self, X, y=None):
        np.random.seed(self.random_seed)

        X_noisy = X.copy().astype(float)
        noise_x = np.random.normal(0, self.noise_std, X_noisy.shape)
        X_noisy = X_noisy + noise_x

        if y is None or not self.noise_on_targets:
            return X_noisy, y

        y_noisy = y.copy().astype(float)
        noise_y = np.random.normal(0, self.noise_std, y_noisy.shape)
        y_noisy = y_noisy + noise_y
        y_noisy = np.rint(y_noisy)
        y_noisy = np.clip(y_noisy, 0, 4).astype(int)

        return X_noisy, y_noisy

    def split_input_output_reshape(self):
        self.feature_columns, self.target_columns = self.get_columns()

        train_X = self.train[self.feature_columns].copy()
        train_y = self.train[self.target_columns].copy()
        test_X = self.test[self.feature_columns].copy()
        test_y = self.test[self.target_columns].copy()

        if self.val is not None:
            val_X = self.val[self.feature_columns].copy()
            val_y = self.val[self.target_columns].copy()

        for col in train_X.columns:
            if str(train_X[col].dtype) == "object":
                train_X[col] = train_X[col].fillna("")
                test_X[col] = test_X[col].fillna("")
                if self.val is not None:
                    val_X[col] = val_X[col].fillna("")
                all_vals = pd.concat([train_X[col], test_X[col]] + ([val_X[col]] if self.val is not None else []), axis=0)
                uniq = sorted(all_vals.astype(str).unique().tolist())
                mapper = {v: i for i, v in enumerate(uniq)}
                train_X[col] = train_X[col].astype(str).map(mapper)
                test_X[col] = test_X[col].astype(str).map(mapper)
                if self.val is not None:
                    val_X[col] = val_X[col].astype(str).map(mapper)
            else:
                train_X[col] = train_X[col].fillna(0)
                test_X[col] = test_X[col].fillna(0)
                if self.val is not None:
                    val_X[col] = val_X[col].fillna(0)

        train_y = train_y.fillna(0).astype(int)
        test_y = test_y.fillna(0).astype(int)
        if self.val is not None:
            val_y = val_y.fillna(0).astype(int)

        self.train_X = self.reshape_input(np.array(train_X))
        self.test_X = self.reshape_input(np.array(test_X))
        self.train_y = self.reshape_output(np.array(train_y))
        self.test_y = self.reshape_output(np.array(test_y))

        if self.val is not None:
            self.val_X = self.reshape_input(np.array(val_X))
            self.val_y = self.reshape_output(np.array(val_y))
        else:
            self.val_X = self.test_X
            self.val_y = self.test_y
            
        if self.inject_noise:
            self.train_X, self.train_y = self.apply_gaussian_noise(self.train_X, self.train_y)
            self.test_X, self.test_y = self.apply_gaussian_noise(self.test_X, self.test_y)

            if self.val is not None:
                self.val_X, self.val_y = self.apply_gaussian_noise(self.val_X, self.val_y)


class Rnn:
    def __init__(self, model_type="lstm", train_x=None, train_y=None, valid_x=None, valid_y=None,
                 num_neurons=None, epoch=100, batch_size=20, lr=0.001,
                 loss='sparse_categorical_crossentropy', activation='tanh',
                 patience=10, class_count=5, prediction_window=3, dropout_rate=0.0):
        self.model_type = model_type.lower()
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activation = activation
        self.patience = patience
        self.class_count = class_count
        self.prediction_window = prediction_window
        self.dropout_rate = dropout_rate
        self.make_model()

    def make_model(self):
        model = Sequential()

        if self.model_type == "simple":
            model.add(SimpleRNN(self.num_neurons[0], input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                return_sequences=(len(self.num_neurons) > 1), activation=self.activation))
        elif self.model_type == "gru":
            model.add(GRU(self.num_neurons[0], input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                          return_sequences=(len(self.num_neurons) > 1), activation=self.activation))
        else:
            model.add(LSTM(self.num_neurons[0], input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                           return_sequences=(len(self.num_neurons) > 1), activation=self.activation))

        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))

        if len(self.num_neurons) > 1:
            for i in range(1, len(self.num_neurons) - 1):
                if self.model_type == "simple":
                    model.add(SimpleRNN(self.num_neurons[i], return_sequences=True, activation=self.activation))
                elif self.model_type == "gru":
                    model.add(GRU(self.num_neurons[i], return_sequences=True, activation=self.activation))
                else:
                    model.add(LSTM(self.num_neurons[i], return_sequences=True, activation=self.activation))
                if self.dropout_rate > 0:
                    model.add(Dropout(self.dropout_rate))

            if self.model_type == "simple":
                model.add(SimpleRNN(self.num_neurons[-1], return_sequences=False, activation=self.activation))
            elif self.model_type == "gru":
                model.add(GRU(self.num_neurons[-1], return_sequences=False, activation=self.activation))
            else:
                model.add(LSTM(self.num_neurons[-1], return_sequences=False, activation=self.activation))

            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.prediction_window * self.class_count, activation='softmax'))
        optimizer = optimizers.Adam(learning_rate=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, log_file_path):
        csv_logger = CSVLogger(log_file_path, append=True, separator=';')
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                     csv_logger]

        y_train = self.train_y
        y_valid = self.valid_y

        if len(y_train.shape) == 2 and y_train.shape[1] == self.prediction_window:
            y_train = np.concatenate([to_categorical(y_train[:, i], num_classes=self.class_count) for i in range(self.prediction_window)], axis=1)
            y_valid = np.concatenate([to_categorical(y_valid[:, i], num_classes=self.class_count) for i in range(self.prediction_window)], axis=1)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Adam(learning_rate=self.lr),
                               metrics=['accuracy'])

        self.history = self.model.fit(self.train_X, y_train,
                                      epochs=self.epoch,
                                      batch_size=self.batch_size,
                                      validation_data=(self.valid_X, y_valid),
                                      verbose=2,
                                      shuffle=True,
                                      callbacks=callbacks)

    def predict(self, X):
        preds = self.model.predict(X)
        preds = preds.reshape((preds.shape[0], self.prediction_window, self.class_count))
        return np.argmax(preds, axis=2)

    def predict_proba(self, X):
        preds = self.model.predict(X)
        preds = preds.reshape((preds.shape[0], self.prediction_window, self.class_count))
        return preds

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

    def load_model(self, model_dir_path, model_id="", segment_number=""):
        if (model_id == "") and (segment_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif segment_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "segment_" + str(segment_number)

        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model