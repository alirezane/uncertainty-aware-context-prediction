import os
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

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None
os.environ['NUMEXPR_MAX_THREADS'] = '30'

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

confidence_threshold = 0.55
ci_threshold = 0.25
entropy_threshold = 1.15

beta = 1.0
lambda_semantic = 0.1

use_restrictions = True
inject_noise = False
noise_std = 0.0
noise_on_targets = False
random_seed = 42

lstm_model_path = "./models/RNN/LSTM/model-1"
fallback_preds_path = "./models/neuro_symbolic/model-1/BNN_Fallback_preds.csv"
tightly_coupled_model_path = "./models/TIGHTLY_COUPLED_BNN/model-" + str(model_number)


def exact_accuracy(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(y_true == y_pred)


def accuracy_at_one(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred) <= 1)


def subset_accuracy(y_true, y_pred, mask):
    mask = np.array(mask).astype(bool)
    if mask.sum() == 0:
        return 0
    return np.mean(np.array(y_true)[mask] == np.array(y_pred)[mask])


def subset_accuracy_at_one(y_true, y_pred, mask):
    mask = np.array(mask).astype(bool)
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs(np.array(y_true)[mask] - np.array(y_pred)[mask]) <= 1)


def prediction_ratio(mask):
    mask = np.array(mask).astype(bool)
    return mask.mean()


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


class TightlyCoupledBnn:
    def __init__(self, train_x=None, train_y=None, valid_x=None, valid_y=None, num_neurons=None,
                 epoch=100, batch_size=20, lr=0.001, activation='tanh', patience=10,
                 class_count=5, prediction_window=3, mc_passes=100,
                 confidence_threshold=0.55, ci_threshold=0.25, entropy_threshold=1.15,
                 beta=1.0, lambda_semantic=0.1, feature_names=None):
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
        self.confidence_threshold = confidence_threshold
        self.ci_threshold = ci_threshold
        self.entropy_threshold = entropy_threshold
        self.beta = beta
        self.lambda_semantic = lambda_semantic
        self.feature_names = feature_names
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

    def load_model(self, model_dir_path):
        predictor_file_path = model_dir_path + "/main_predictor"
        with open(predictor_file_path + ".json", 'r') as json_file:
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
        self.model.load_weights(predictor_file_path + ".weights.h5")

    def predict_selective(self, X):
        all_probs = []
        for i in range(self.mc_passes):
            logits = self.model(X, training=True).numpy()
            logits = logits.reshape((logits.shape[0], self.prediction_window, self.class_count))
            probs = tf.nn.softmax(logits, axis=2).numpy()
            all_probs.append(probs)

        mc_probs = np.array(all_probs)
        mean_probs = np.mean(mc_probs, axis=0)
        std_probs = np.std(mc_probs, axis=0)
        preds = np.argmax(mean_probs, axis=2)
        confidence = np.max(mean_probs, axis=2)

        ci_width = np.zeros((mean_probs.shape[0], mean_probs.shape[1]))
        entropy = np.zeros((mean_probs.shape[0], mean_probs.shape[1]))
        accepted_mask = np.zeros((mean_probs.shape[0], mean_probs.shape[1]), dtype=int)

        for i in range(mean_probs.shape[0]):
            for j in range(mean_probs.shape[1]):
                pred_class = preds[i, j]
                ci_width[i, j] = 1.96 * std_probs[i, j, pred_class]
                entropy[i, j] = -np.sum(mean_probs[i, j, :] * np.log(mean_probs[i, j, :] + 1e-10))
                accepted = (
                    confidence[i, j] >= self.confidence_threshold and
                    ci_width[i, j] <= self.ci_threshold and
                    entropy[i, j] <= self.entropy_threshold
                )
                accepted_mask[i, j] = 1 if accepted else 0

        selective_preds = preds.copy()
        selective_preds[accepted_mask == 0] = -1
        return selective_preds, preds, accepted_mask, confidence, ci_width, entropy


class LstmModel:
    def __init__(self, train_x=None, train_y=None, valid_x=None, valid_y=None, num_neurons=None,
                 epoch=100, batch_size=20, lr=0.001,
                 loss='categorical_crossentropy', activation='tanh',
                 patience=10, class_count=5, prediction_window=3, dropout_rate=0.0):
        self.model_type = "lstm"
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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(self.num_neurons[0], input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                       return_sequences=(len(self.num_neurons) > 1), activation=self.activation))
        if len(self.num_neurons) > 1:
            for i in range(1, len(self.num_neurons) - 1):
                model.add(tf.keras.layers.LSTM(self.num_neurons[i], return_sequences=True, activation=self.activation))
            model.add(tf.keras.layers.LSTM(self.num_neurons[-1], return_sequences=False, activation=self.activation))
        model.add(tf.keras.layers.Dense(self.prediction_window * self.class_count, activation='softmax'))
        self.model = model

    def load_model(self, model_dir_path):
        predictor_file_path = model_dir_path + "/main_predictor"
        with open(predictor_file_path + ".json", 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(predictor_file_path + ".h5")

    def predict(self, X):
        preds = self.model.predict(X)
        preds = preds.reshape((preds.shape[0], self.prediction_window, self.class_count))
        return np.argmax(preds, axis=2)


if __name__ == '__main__':
    try:
        output_dir = util_classification.make_model_dir("./models/TIGHTLY_COUPLED_BNN", "evaluate_tightly_coupled")

        dataset_rnn = util_classification.GenerateDate(data_file_path=data_file_path,
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
                                                       use_restrictions=use_restrictions,
                                                       inject_noise=inject_noise,
                                                       noise_std=noise_std,
                                                       noise_on_targets=noise_on_targets,
                                                       random_seed=random_seed)

        dataset_tab = util_classification.GenerateDate(data_file_path=data_file_path,
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
                                                       use_restrictions=use_restrictions,
                                                       inject_noise=inject_noise,
                                                       noise_std=noise_std,
                                                       noise_on_targets=noise_on_targets,
                                                       random_seed=random_seed)

        tightly_model = TightlyCoupledBnn(train_x=dataset_rnn.train_X,
                                          train_y=dataset_rnn.train_y,
                                          valid_x=dataset_rnn.val_X,
                                          valid_y=dataset_rnn.val_y,
                                          num_neurons=main_num_neurons,
                                          epoch=main_epoch,
                                          batch_size=main_batch_size,
                                          lr=main_lr,
                                          activation=main_activation,
                                          patience=main_patience,
                                          class_count=class_count,
                                          prediction_window=prediction_window,
                                          mc_passes=mc_passes,
                                          confidence_threshold=confidence_threshold,
                                          ci_threshold=ci_threshold,
                                          entropy_threshold=entropy_threshold,
                                          beta=beta,
                                          lambda_semantic=lambda_semantic,
                                          feature_names=dataset_rnn.feature_columns)
        tightly_model.load_model(tightly_coupled_model_path)

        selective_preds, forced_preds, accepted_mask, confidence, ci_width, entropy = tightly_model.predict_selective(dataset_rnn.test_X)

        lstm = LstmModel(train_x=dataset_rnn.train_X,
                         train_y=dataset_rnn.train_y,
                         valid_x=dataset_rnn.val_X,
                         valid_y=dataset_rnn.val_y,
                         num_neurons=main_num_neurons,
                         epoch=main_epoch,
                         batch_size=main_batch_size,
                         lr=main_lr,
                         activation=main_activation,
                         patience=main_patience,
                         class_count=class_count,
                         prediction_window=prediction_window)
        lstm.load_model(lstm_model_path)
        lstm_preds = lstm.predict(dataset_rnn.test_X)

        fallback_df = pd.read_csv(fallback_preds_path)

        output_lines = []

        for i in range(prediction_window):
            y_true = dataset_rnn.test[f"target_class_t+{i + 1}"].astype(int).values
            tight_forced = forced_preds[:, i]
            tight_selective = selective_preds[:, i]
            accept = accepted_mask[:, i].astype(bool)
            reject = ~accept

            lstm_pred = lstm_preds[:, i]
            fb_col = f"BNN_Fallback_target_{i + 1}"
            if fb_col in fallback_df.columns:
                fallback_pred = fallback_df[fb_col].fillna(-1).astype(int).values
            else:
                fallback_pred = np.full_like(y_true, -1)

            forced_acc = exact_accuracy(y_true, tight_forced)
            selective_acc = subset_accuracy(y_true, tight_forced, accept)
            rejected_acc = subset_accuracy(y_true, tight_forced, reject)
            pr = prediction_ratio(accept)

            lstm_acc_accept = subset_accuracy(y_true, lstm_pred, accept)
            lstm_acc_reject = subset_accuracy(y_true, lstm_pred, reject)
            fb_acc_accept = subset_accuracy(y_true, fallback_pred, accept)
            fb_acc_reject = subset_accuracy(y_true, fallback_pred, reject)

            output_lines.append(f"Prediction Window {i + 1}")
            output_lines.append(f"Tightly Coupled Forced Accuracy: {forced_acc}")
            output_lines.append(f"Tightly Coupled Selective Accuracy: {selective_acc}")
            output_lines.append(f"Tightly Coupled Prediction Ratio: {pr}")
            output_lines.append(f"Tightly Coupled Rejected Accuracy: {rejected_acc}")
            output_lines.append(f"LSTM Accuracy on Accepted Subset: {lstm_acc_accept}")
            output_lines.append(f"LSTM Accuracy on Rejected Subset: {lstm_acc_reject}")
            output_lines.append(f"BNN_Fallback Accuracy on Accepted Subset: {fb_acc_accept}")
            output_lines.append(f"BNN_Fallback Accuracy on Rejected Subset: {fb_acc_reject}")
            output_lines.append("")

            dataset_rnn.test[f"TC_BNN_target_{i + 1}"] = tight_selective
            dataset_rnn.test[f"TC_BNN_forced_target_{i + 1}"] = tight_forced
            dataset_rnn.test[f"TC_BNN_accept_t+{i + 1}"] = accepted_mask[:, i]
            dataset_rnn.test[f"TC_BNN_confidence_t+{i + 1}"] = confidence[:, i]
            dataset_rnn.test[f"TC_BNN_ci_width_t+{i + 1}"] = ci_width[:, i]
            dataset_rnn.test[f"TC_BNN_entropy_t+{i + 1}"] = entropy[:, i]
            dataset_rnn.test[f"LSTM_target_{i + 1}"] = lstm_pred
            dataset_rnn.test[f"BNN_Fallback_target_{i + 1}"] = fallback_pred

            logger.debug(f"Prediction Window {i + 1}")
            logger.debug(f"Tightly Coupled Forced Accuracy: {forced_acc}")
            logger.debug(f"Tightly Coupled Selective Accuracy: {selective_acc}")
            logger.debug(f"Tightly Coupled Prediction Ratio: {pr}")
            logger.debug(f"Tightly Coupled Rejected Accuracy: {rejected_acc}")
            logger.debug(f"LSTM Accuracy on Accepted Subset: {lstm_acc_accept}")
            logger.debug(f"LSTM Accuracy on Rejected Subset: {lstm_acc_reject}")
            logger.debug(f"BNN_Fallback Accuracy on Accepted Subset: {fb_acc_accept}")
            logger.debug(f"BNN_Fallback Accuracy on Rejected Subset: {fb_acc_reject}")

        with open(output_dir + "/results.txt", "w") as f:
            f.write("\n".join(output_lines))

        dataset_rnn.test.to_csv(output_dir + "/tightly_coupled_evaluation.csv", index=False)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)