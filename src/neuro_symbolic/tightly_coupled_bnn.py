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

model_type = "TIGHTLY_COUPLED_BNN"

log_file_path = './logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower())
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

confidence_threshold = 0.55
ci_threshold = 0.25
entropy_threshold = 1.15

beta = 1.0
lambda_semantic = 0.1
training_data_fraction = 1.0

use_restrictions = True
inject_noise = False
noise_std = 0.0
noise_on_targets = False
random_seed = 42


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
        self.semantic_indices = self.get_semantic_feature_indices()
        self.make_model()

    def get_semantic_feature_indices(self):
        if self.feature_names is None:
            return []
        semantic_names = [x for x in self.feature_names if (
            "restriction" in x.lower() or
            "typedesc" in x.lower() or
            "description" in x.lower() or
            "duration" in x.lower() or
            "effectiveonph" in x.lower() or
            "meter_ratio" in x.lower() or
            "disabled_ratio" in x.lower() or
            "loading_ratio" in x.lower() or
            "permit_ratio" in x.lower() or
            "clearway_ratio" in x.lower() or
            "no_parking_ratio" in x.lower() or
            "no_stopping_ratio" in x.lower() or
            "active_restriction_ratio" in x.lower() or
            "restricted_bay_count" in x.lower()
        )]
        return [self.feature_names.index(x) for x in semantic_names if x in self.feature_names]

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

    def semantic_loss(self, x_input, logits):
        if len(self.semantic_indices) == 0:
            return 0.0

        flat_x = tf.reshape(x_input, (tf.shape(x_input)[0], -1))
        semantic_cols = []
        for idx in self.semantic_indices:
            if idx < flat_x.shape[1]:
                semantic_cols.append(flat_x[:, idx])

        if len(semantic_cols) == 0:
            return 0.0

        semantic_signal = tf.add_n(semantic_cols) / float(len(semantic_cols))
        semantic_signal = tf.cast(semantic_signal, tf.float32)

        probs = tf.nn.softmax(tf.reshape(logits, (-1, self.prediction_window, self.class_count)), axis=2)
        high_occ_prob = probs[:, :, 3] + probs[:, :, 4]
        low_occ_prob = probs[:, :, 0] + probs[:, :, 1]

        semantic_signal = tf.expand_dims(semantic_signal, axis=1)
        penalty_high = tf.maximum(0.0, semantic_signal - high_occ_prob)
        penalty_low = tf.maximum(0.0, low_occ_prob - (1.0 - semantic_signal))

        return tf.reduce_mean(penalty_high + penalty_low)

    def loss_fn(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=self.class_count)
        y_true_one_hot = tf.reshape(y_true_one_hot, (-1, self.prediction_window * self.class_count))
        probs = tf.nn.softmax(y_pred)
        ce = tf.keras.losses.categorical_crossentropy(y_true_one_hot, probs, from_logits=False)
        kl = tf.add_n(self.model.losses) if len(self.model.losses) > 0 else 0.0
        sem = self.semantic_loss(self.current_batch_x, y_pred)
        return ce + self.beta * kl + self.lambda_semantic * sem

    def fit(self, log_file_path):
        csv_logger = tf.keras.callbacks.CSVLogger(log_file_path, append=True, separator=';')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),
                     tf.keras.callbacks.ModelCheckpoint(filepath='best_model.weights.h5',
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        save_weights_only=True),
                     csv_logger]

        class SemanticModel(tf.keras.Model):
            def __init__(self, outer_model):
                super().__init__()
                self.outer_model = outer_model
                self.inner_model = outer_model.model

            def train_step(self, data):
                x, y = data
                self.outer_model.current_batch_x = x
                with tf.GradientTape() as tape:
                    y_pred = self.inner_model(x, training=True)
                    loss = self.outer_model.loss_fn(y, y_pred)
                grads = tape.gradient(loss, self.inner_model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.inner_model.trainable_variables))
                return {"loss": loss}

            def test_step(self, data):
                x, y = data
                self.outer_model.current_batch_x = x
                y_pred = self.inner_model(x, training=False)
                loss = self.outer_model.loss_fn(y, y_pred)
                return {"loss": loss}

            def call(self, inputs, training=False):
                return self.inner_model(inputs, training=training)

        self.train_wrapper = SemanticModel(self)
        self.train_wrapper.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

        self.history = self.train_wrapper.fit(self.train_X, self.train_y,
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

    def predict_selective(self, X):
        mc_probs = self.predict_mc_probabilities(X)
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

        return selective_preds, preds, mean_probs, confidence, ci_width, entropy, accepted_mask

    def save_model(self, model_dir_path):
        predictor_file_path = model_dir_path + "/main_predictor"
        model_json = self.model.to_json()
        with open(predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(predictor_file_path + ".weights.h5")

        hist_df = pd.DataFrame(self.history.history)
        with open(predictor_file_path + '_history.csv', 'wb') as file_pi:
            hist_df.to_csv(file_pi)

        params = {
            "prediction_window": self.prediction_window,
            "class_count": self.class_count,
            "mc_passes": self.mc_passes,
            "num_neurons": self.num_neurons,
            "confidence_threshold": self.confidence_threshold,
            "ci_threshold": self.ci_threshold,
            "entropy_threshold": self.entropy_threshold,
            "beta": self.beta,
            "lambda_semantic": self.lambda_semantic
        }
        with open(predictor_file_path + "_params.pkl", "wb") as f:
            pickle.dump(params, f)

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
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss=self.loss_fn)
        self.model.load_weights(predictor_file_path + ".weights.h5")


if __name__ == '__main__':
    try:
        if not infer:
            model_number = 1
            model_dir_path = util_classification.make_model_dir("./models/TIGHTLY_COUPLED_BNN", "tightly_coupled_bnn")
        else:
            model_dir_path = "./models/TIGHTLY_COUPLED_BNN/model-" + str(model_number)

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
                                                   use_restrictions=use_restrictions,
                                                   inject_noise=inject_noise,
                                                   noise_std=noise_std,
                                                   noise_on_targets=noise_on_targets,
                                                   random_seed=random_seed)

        if training_data_fraction < 1.0:
            sample_size = int(dataset.train_X.shape[0] * training_data_fraction)
            if sample_size > 0:
                dataset.train_X = dataset.train_X[:sample_size]
                dataset.train_y = dataset.train_y[:sample_size]

        predictor = TightlyCoupledBnn(train_x=dataset.train_X,
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
                                      confidence_threshold=confidence_threshold,
                                      ci_threshold=ci_threshold,
                                      entropy_threshold=entropy_threshold,
                                      beta=beta,
                                      lambda_semantic=lambda_semantic,
                                      feature_names=dataset.feature_columns)

        if not infer:
            logger.debug(f"main_num_neurons = {main_num_neurons}")
            logger.debug(f"main_epoch = {main_epoch}")
            logger.debug(f"main_batch_size = {main_batch_size}")
            logger.debug(f"main_lr = {main_lr}")
            logger.debug(f"main_activation = {main_activation}")
            logger.debug(f"main_patience = {main_patience}")
            logger.debug(f"prediction_window = {prediction_window}")
            logger.debug(f"class_count = {class_count}")
            logger.debug(f"mc_passes = {mc_passes}")
            logger.debug(f"confidence_threshold = {confidence_threshold}")
            logger.debug(f"ci_threshold = {ci_threshold}")
            logger.debug(f"entropy_threshold = {entropy_threshold}")
            logger.debug(f"beta = {beta}")
            logger.debug(f"lambda_semantic = {lambda_semantic}")

            t = time.time()
            predictor.fit(log_file_path)
            elapsed = time.time() - t
            logger.debug('Elapsed: {}'.format(elapsed))

            predictor.save_model(model_dir_path)

        else:
            predictor.load_model(model_dir_path)

            selective_preds, forced_preds, mean_probs, confidence, ci_width, entropy, accepted_mask = predictor.predict_selective(dataset.test_X)

            for i in range(prediction_window):
                dataset.test[f"TC_BNN_target_{i + 1}"] = selective_preds[:, i]
                dataset.test[f"TC_BNN_forced_target_{i + 1}"] = forced_preds[:, i]
                dataset.test[f"TC_BNN_confidence_t+{i + 1}"] = confidence[:, i]
                dataset.test[f"TC_BNN_ci_width_t+{i + 1}"] = ci_width[:, i]
                dataset.test[f"TC_BNN_entropy_t+{i + 1}"] = entropy[:, i]
                dataset.test[f"TC_BNN_accept_t+{i + 1}"] = accepted_mask[:, i]

                for j in range(class_count):
                    dataset.test[f"TC_BNN_mean_prob_t+{i + 1}_class_{j}"] = mean_probs[:, i, j]

            dataset.test.to_csv(model_dir_path + "/TC_BNN_preds.csv", index=False)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)