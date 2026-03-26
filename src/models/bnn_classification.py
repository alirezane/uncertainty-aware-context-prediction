import os
import time
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

import util_classification

tfd = tfp.distributions
tfpl = tfp.layers

model_type = "BNN"

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
main_activation = "tanh"
main_patience = 10
class_count = 5
mc_passes = 100
threshold = None
training_data_fraction = 1.0


def accuracy_at_one(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    valid = y_pred >= 0
    if valid.sum() == 0:
        return 0
    return np.mean(np.abs(y_true[valid] - y_pred[valid]) <= 1)


def exact_accuracy(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    valid = y_pred >= 0
    if valid.sum() == 0:
        return 0
    return np.mean(y_true[valid] == y_pred[valid])


def prediction_ratio(y_pred):
    y_pred = np.array(y_pred).flatten()
    return np.mean(y_pred >= 0)


def prior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1.0),
            reinterpreted_batch_ndims=1))
    ])


def posterior(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.IndependentNormal(n)
    ])


class Bnn:
    def __init__(self, train_x=None, train_y=None, valid_x=None, valid_y=None, num_neurons=None,
                 epoch=100, batch_size=20, lr=0.001, activation='tanh', patience=10,
                 class_count=5, prediction_window=3, mc_passes=100, threshold=None):
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.activation = activation
        self.patience = patience
        self.class_count = class_count
        self.prediction_window = prediction_window
        self.mc_passes = mc_passes
        self.threshold = threshold
        self.kl_weight = 1.0 / max(self.train_X.shape[0], 1)
        self.make_model()

    def make_model(self):
        inputs = tf.keras.Input(shape=(self.train_X.shape[1], self.train_X.shape[2]))
        x = tf.keras.layers.Flatten()(inputs)

        for n in self.num_neurons:
            x = tfpl.DenseVariational(units=n,
                                      make_prior_fn=prior,
                                      make_posterior_fn=posterior,
                                      kl_weight=self.kl_weight,
                                      activation=self.activation)(x)

        outputs = tfpl.DenseVariational(units=self.prediction_window * self.class_count,
                                        make_prior_fn=prior,
                                        make_posterior_fn=posterior,
                                        kl_weight=self.kl_weight,
                                        activation=None)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss=self.loss_fn)

    def loss_fn(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.class_count)
        y_true_one_hot = tf.reshape(y_true_one_hot, (-1, self.prediction_window * self.class_count))
        probs = tf.nn.softmax(y_pred)
        ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, probs, from_logits=False)
        return ce

    def fit(self, log_file_path):
        csv_logger = tf.keras.callbacks.CSVLogger(log_file_path, append=True, separator=';')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),
                     tf.keras.callbacks.ModelCheckpoint(filepath='best_model.weights.h5',
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        save_weights_only=True),
                     csv_logger]

        self.history = self.model.fit(self.train_X, self.train_y,
                                      epochs=self.epoch,
                                      batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y),
                                      verbose=2,
                                      shuffle=True,
                                      callbacks=callbacks)

    def predict_mc_probabilities(self, X):
        all_probs = []
        for i in range(self.mc_passes):
            logits = self.model(X, training=True).numpy()
            logits = logits.reshape((logits.shape[0], self.prediction_window, self.class_count))
            probs = tf.nn.softmax(logits, axis=2).numpy()
            all_probs.append(probs)
        return np.array(all_probs)

    def predict_with_threshold(self, X):
        mc_probs = self.predict_mc_probabilities(X)
        median_probs = np.median(mc_probs, axis=0)
        preds = np.argmax(median_probs, axis=2)
        max_probs = np.max(median_probs, axis=2)

        if self.threshold is None:
            return preds, median_probs, max_probs

        threshold_preds = preds.copy()
        for i in range(threshold_preds.shape[0]):
            for j in range(threshold_preds.shape[1]):
                passed = np.where(median_probs[i, j, :] >= self.threshold)[0]
                if len(passed) != 1:
                    threshold_preds[i, j] = -1
                else:
                    threshold_preds[i, j] = int(passed[0])

        return threshold_preds, median_probs, max_probs

    def save_model(self, model_dir_path):
        self.predictor_file_path = model_dir_path + "/main_predictor"
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".weights.h5")

        hist_df = pd.DataFrame(self.history.history)
        with open(self.predictor_file_path + '_history.csv', 'wb') as file_pi:
            hist_df.to_csv(file_pi)

        params = {
            "prediction_window": self.prediction_window,
            "class_count": self.class_count,
            "mc_passes": self.mc_passes,
            "threshold": self.threshold,
            "num_neurons": self.num_neurons
        }
        with open(self.predictor_file_path + "_params.pkl", "wb") as f:
            pickle.dump(params, f)

    def load_model(self, model_dir_path):
        self.predictor_file_path = model_dir_path + "/main_predictor"
        with open(self.predictor_file_path + ".json", 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = tf.keras.models.model_from_json(
            loaded_model_json,
            custom_objects={
                "DenseVariational": tfpl.DenseVariational,
                "DistributionLambda": tfpl.DistributionLambda,
                "VariableLayer": tfpl.VariableLayer,
                "IndependentNormal": tfpl.IndependentNormal,
                "prior": prior,
                "posterior": posterior
            }
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss=self.loss_fn)
        self.model.load_weights(self.predictor_file_path + ".weights.h5")


if __name__ == '__main__':
    try:
        if threshold is None:
            model_folder = "./models/BNN"
            code_name = "bnn_classification"
            label_name = "BNN"
        elif threshold == 0.2:
            model_folder = "./models/BNN_20"
            code_name = "bnn_classification"
            label_name = "BNN20"
        elif threshold == 0.3:
            model_folder = "./models/BNN_30"
            code_name = "bnn_classification"
            label_name = "BNN30"
        else:
            model_folder = "./models/BNN"
            code_name = "bnn_classification"
            label_name = "BNN"

        if not infer:
            model_number = 1
            model_dir_path = util_classification.make_model_dir(model_folder, code_name)
        else:
            model_dir_path = model_folder + "/model-" + str(model_number)

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

        if training_data_fraction < 1.0:
            sample_size = int(dataset.train_X.shape[0] * training_data_fraction)
            if sample_size > 0:
                dataset.train_X = dataset.train_X[:sample_size]
                dataset.train_y = dataset.train_y[:sample_size]

        if not infer:
            logger.debug(f"Creating model with specs:")
            logger.debug(f"main_num_neurons = {main_num_neurons}")
            logger.debug(f"main_epoch = {main_epoch}")
            logger.debug(f"main_batch_size = {main_batch_size}")
            logger.debug(f"main_lr = {main_lr}")
            logger.debug(f"main_activation = {main_activation}")
            logger.debug(f"main_patience = {main_patience}")
            logger.debug(f"prediction_window = {prediction_window}")
            logger.debug(f"class_count = {class_count}")
            logger.debug(f"mc_passes = {mc_passes}")
            logger.debug(f"threshold = {threshold}")

            predictor = Bnn(train_x=dataset.train_X,
                            train_y=dataset.train_y,
                            valid_x=dataset.val_X,
                            valid_y=dataset.val_y,
                            num_neurons=main_num_neurons,
                            epoch=main_epoch,
                            batch_size=main_batch_size,
                            lr=main_lr,
                            activation=main_activation,
                            patience=main_patience,
                            class_count=class_count,
                            prediction_window=prediction_window,
                            mc_passes=mc_passes,
                            threshold=threshold)

            logger.debug("Training main predictor")
            t = time.time()
            predictor.fit(log_file_path)
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

            predictor.save_model(model_dir_path)

        else:
            predictor = Bnn(train_x=dataset.train_X,
                            train_y=dataset.train_y,
                            valid_x=dataset.val_X,
                            valid_y=dataset.val_y,
                            num_neurons=main_num_neurons,
                            epoch=main_epoch,
                            batch_size=main_batch_size,
                            lr=main_lr,
                            activation=main_activation,
                            patience=main_patience,
                            class_count=class_count,
                            prediction_window=prediction_window,
                            mc_passes=mc_passes,
                            threshold=threshold)
            predictor.load_model(model_dir_path)

            main_y_hat, median_probs, max_probs = predictor.predict_with_threshold(dataset.test_X)

            dataset.test = util_classification.add_preds(dataset.test, main_y_hat, label_name)

            for i in range(prediction_window):
                dataset.test[f"{label_name}_confidence_t+{i + 1}"] = max_probs[:, i]

            for i in range(prediction_window):
                for j in range(class_count):
                    dataset.test[f"{label_name}_median_prob_t+{i + 1}_class_{j}"] = median_probs[:, i, j]

            target_cols = [f"target_class_t+{i + 1}" for i in range(prediction_window)]
            pred_cols = [f"{label_name}_target_{i + 1}" for i in range(prediction_window)]

            output_lines = []
            for i in range(prediction_window):
                y_true = dataset.test[target_cols[i]].astype(int).values
                y_pred = dataset.test[pred_cols[i]].fillna(-1).astype(int).values

                pr = prediction_ratio(y_pred)
                acc = exact_accuracy(y_true, y_pred)
                acc1 = accuracy_at_one(y_true, y_pred)

                output_lines.append(f"Prediction Window {i + 1}")
                output_lines.append(f"Prediction Ratio: {pr}")
                output_lines.append(f"Accuracy: {acc}")
                output_lines.append(f"Accuracy@1: {acc1}")
                output_lines.append("")

                logger.debug(f"{label_name} - Prediction Window {i + 1}")
                logger.debug(f"Prediction Ratio: {pr}")
                logger.debug(f"Accuracy: {acc}")
                logger.debug(f"Accuracy@1: {acc1}")

            with open(model_dir_path + "/results.txt", "w") as f:
                f.write("\n".join(output_lines))

            dataset.test.to_csv(model_dir_path + f"/{label_name}_preds.csv", index=False)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)