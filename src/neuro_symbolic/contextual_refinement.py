import os
import re
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
threshold = 0.3

rules_model_number = 1
rules_dir_path = "./models/neuro_symbolic/model-" + str(rules_model_number)

use_restrictions = True
inject_noise = False
noise_std = 0.0
noise_on_targets = False
random_seed = 42


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


def convert_value(x):
    try:
        if pd.isna(x):
            return None
    except BaseException:
        pass

    if isinstance(x, (int, float)):
        return x

    x = str(x).strip()
    if x == "":
        return None

    try:
        return float(x)
    except BaseException:
        return x


def evaluate_condition(row, cond):
    cond = cond.strip()
    cond = cond.replace("[", "").replace("]", "")
    m = re.match(r"^\((.+?)\s*(<=|>=|<|>|==|!=)\s*(.+)\)$", cond)
    if m is None:
        return False

    feature_name = m.group(1).strip()
    operator = m.group(2).strip()
    threshold_value = convert_value(m.group(3).strip())

    if feature_name not in row.index:
        return False

    value = row[feature_name]

    try:
        if pd.isna(value):
            return False
    except BaseException:
        pass

    try:
        value = float(value)
        threshold_value = float(threshold_value)

        if operator == "<=":
            return value <= threshold_value
        if operator == ">=":
            return value >= threshold_value
        if operator == "<":
            return value < threshold_value
        if operator == ">":
            return value > threshold_value
        if operator == "==":
            return value == threshold_value
        if operator == "!=":
            return value != threshold_value
        return False
    except BaseException:
        value = str(value).strip()
        threshold_value = str(threshold_value).strip()

        if operator == "==":
            return value == threshold_value
        if operator == "!=":
            return value != threshold_value
        return False


def evaluate_rule(row, rule_text):
    if pd.isna(rule_text):
        return False

    rule_text = str(rule_text).strip()
    if rule_text == "" or rule_text.lower() == "true":
        return True

    conditions = [x.strip() for x in rule_text.split(" and ")]
    for cond in conditions:
        if not evaluate_condition(row, cond):
            return False
    return True


def get_admissible_classes(df, rules_df, target_number):
    admissible_sets = []
    plp_preds = []
    matched_rule_ids = []
    matched_supports = []

    target_rules = rules_df[rules_df["target_number"] == target_number].copy()
    target_rules.sort_values(["sample_count", "rule_id"], ascending=[False, True], inplace=True)

    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        matched_classes = []
        matched_rule_id = None
        matched_support = None

        for _, rule_row in target_rules.iterrows():
            if evaluate_rule(row, rule_row["rule"]):
                pred_class = int(rule_row["predicted_class"])
                if pred_class not in matched_classes:
                    matched_classes.append(pred_class)
                if matched_rule_id is None:
                    matched_rule_id = int(rule_row["rule_id"])
                    matched_support = int(rule_row["sample_count"])

        matched_classes = sorted(list(set(matched_classes)))

        if len(matched_classes) == 0:
            admissible_sets.append([0, 1, 2, 3, 4])
            plp_preds.append(-1)
            matched_rule_ids.append(None)
            matched_supports.append(None)
        else:
            admissible_sets.append(matched_classes)
            plp_preds.append(matched_classes[0])
            matched_rule_ids.append(matched_rule_id)
            matched_supports.append(matched_support)

    return admissible_sets, np.array(plp_preds), matched_rule_ids, matched_supports


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


if __name__ == '__main__':
    try:
        model_dir_path = util_classification.make_model_dir("./models/neuro_symbolic", "contextual_refinement")

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

        rules_frames = []
        for target_number in range(1, prediction_window + 1):
            file_path = rules_dir_path + f"/decision_tree_rules_target_{target_number}.csv"
            if os.path.exists(file_path):
                tmp = pd.read_csv(file_path)
                rules_frames.append(tmp)

        if len(rules_frames) == 0:
            raise Exception("No decision tree rules files found")

        rules_df = pd.concat(rules_frames, axis=0).reset_index(drop=True)

        bnn_predictor = Bnn(train_x=dataset.train_X,
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

        bnn_model_dir_path = "./models/BNN_30/model-" + str(model_number) if threshold == 0.3 else "./models/BNN_20/model-" + str(model_number)
        if threshold is None:
            bnn_model_dir_path = "./models/BNN/model-" + str(model_number)

        logger.debug("Loading BNN model")
        bnn_predictor.load_model(bnn_model_dir_path)

        bnn_preds, median_probs, max_probs = bnn_predictor.predict_with_threshold(dataset.test_X)

        output_lines = []

        for target_number in range(1, prediction_window + 1):
            admissible_sets, plp_preds, matched_rule_ids, matched_supports = get_admissible_classes(dataset.test,
                                                                                                     rules_df,
                                                                                                     target_number)

            dataset.test[f"BNN_target_{target_number}"] = bnn_preds[:, target_number - 1]
            dataset.test[f"BNN_confidence_t+{target_number}"] = max_probs[:, target_number - 1]
            dataset.test[f"PLP_target_{target_number}"] = plp_preds
            dataset.test[f"PLP_rule_id_t+{target_number}"] = matched_rule_ids
            dataset.test[f"PLP_rule_support_t+{target_number}"] = matched_supports

            refined_preds = []
            refined_confidences = []
            refined_choice_source = []

            for i in range(dataset.test.shape[0]):
                raw_pred = int(bnn_preds[i, target_number - 1])
                raw_conf = float(max_probs[i, target_number - 1])
                probs = median_probs[i, target_number - 1, :]
                admissible = admissible_sets[i]

                if raw_pred >= 0:
                    refined_preds.append(raw_pred)
                    refined_confidences.append(raw_conf)
                    refined_choice_source.append("BNN")
                    continue

                admissible = [c for c in admissible if 0 <= c < class_count]

                if len(admissible) == 0:
                    refined_preds.append(int(plp_preds[i]))
                    refined_confidences.append(-1.0)
                    refined_choice_source.append("PLP")
                    continue

                admissible_probs = probs[admissible]
                prob_sum = admissible_probs.sum()

                if prob_sum <= 0:
                    refined_preds.append(int(plp_preds[i]))
                    refined_confidences.append(-1.0)
                    refined_choice_source.append("PLP")
                    continue

                renorm_probs = admissible_probs / prob_sum
                best_local_idx = int(np.argmax(renorm_probs))
                refined_class = int(admissible[best_local_idx])
                refined_conf = float(renorm_probs[best_local_idx])

                if refined_conf >= threshold:
                    refined_preds.append(refined_class)
                    refined_confidences.append(refined_conf)
                    refined_choice_source.append("REFINED_BNN")
                else:
                    refined_preds.append(int(plp_preds[i]))
                    refined_confidences.append(refined_conf)
                    refined_choice_source.append("PLP")

            dataset.test[f"BNN_Refined_target_{target_number}"] = refined_preds
            dataset.test[f"BNN_Refined_confidence_t+{target_number}"] = refined_confidences
            dataset.test[f"BNN_Refined_source_t+{target_number}"] = refined_choice_source
            dataset.test[f"BNN_Refined_admissible_t+{target_number}"] = [str(x) for x in admissible_sets]

            y_true = dataset.test[f"target_class_t+{target_number}"].astype(int).values
            y_pred = dataset.test[f"BNN_Refined_target_{target_number}"].fillna(-1).astype(int).values

            pr = prediction_ratio(y_pred)
            acc = exact_accuracy(y_true, y_pred)
            acc1 = accuracy_at_one(y_true, y_pred)

            output_lines.append(f"Prediction Window {target_number}")
            output_lines.append(f"Prediction Ratio: {pr}")
            output_lines.append(f"Accuracy: {acc}")
            output_lines.append(f"Accuracy@1: {acc1}")
            output_lines.append("")

            logger.debug(f"BNN + Contextual Refinement - Prediction Window {target_number}")
            logger.debug(f"Prediction Ratio: {pr}")
            logger.debug(f"Accuracy: {acc}")
            logger.debug(f"Accuracy@1: {acc1}")

        with open(model_dir_path + "/results.txt", "w") as f:
            f.write("\n".join(output_lines))

        dataset.test.to_csv(model_dir_path + "/BNN_Refined_preds.csv", index=False)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)