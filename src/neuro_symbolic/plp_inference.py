import os
import re
import ast
import logging
import datetime
import pandas as pd

import util_classification

logging.basicConfig(filename='./logs/{}.log'.format(os.path.basename(__file__).split('.')[0].lower()),
                    level=logging.DEBUG,
                    format='%(asctime)s : %(name)s : %(message)s',
                    filemode='w')
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None


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

rules_model_number = 1
rules_dir_path = "./models/neuro_symbolic/model-" + str(rules_model_number)

use_restrictions = True
inject_noise = False
noise_std = 0.0
noise_on_targets = False
random_seed = 42


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
        return ast.literal_eval(x)
    except BaseException:
        pass

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
    threshold_raw = m.group(3).strip()

    if feature_name not in row.index:
        return False

    value = row[feature_name]
    threshold = convert_value(threshold_raw)

    try:
        if pd.isna(value):
            return False
    except BaseException:
        pass

    try:
        value = float(value)
        threshold = float(threshold)

        if operator == "<=":
            return value <= threshold
        if operator == ">=":
            return value >= threshold
        if operator == "<":
            return value < threshold
        if operator == ">":
            return value > threshold
        if operator == "==":
            return value == threshold
        if operator == "!=":
            return value != threshold
        return False
    except BaseException:
        value = str(value).strip()
        threshold = str(threshold).strip()

        if operator == "==":
            return value == threshold
        if operator == "!=":
            return value != threshold
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


def apply_rules_to_dataset(df, rules_df, target_number):
    preds = []
    matched_rule_ids = []
    matched_supports = []

    target_rules = rules_df[rules_df["target_number"] == target_number].copy()
    target_rules.sort_values(["sample_count", "rule_id"], ascending=[False, True], inplace=True)

    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        pred = -1
        matched_rule_id = None
        matched_support = None

        for _, rule_row in target_rules.iterrows():
            if evaluate_rule(row, rule_row["rule"]):
                pred = int(rule_row["predicted_class"])
                matched_rule_id = int(rule_row["rule_id"])
                matched_support = int(rule_row["sample_count"])
                break

        preds.append(pred)
        matched_rule_ids.append(matched_rule_id)
        matched_supports.append(matched_support)

    return preds, matched_rule_ids, matched_supports


def prediction_ratio(y_pred):
    y_pred = pd.Series(y_pred)
    return (y_pred >= 0).mean()


def exact_accuracy(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    valid = y_pred >= 0
    if valid.sum() == 0:
        return 0
    return (y_true[valid] == y_pred[valid]).mean()


def accuracy_at_one(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    valid = y_pred >= 0
    if valid.sum() == 0:
        return 0
    return ((y_true[valid] - y_pred[valid]).abs() <= 1).mean()


if __name__ == '__main__':
    try:
        model_dir_path = util_classification.make_model_dir("./models/neuro_symbolic", "plp_inference")

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

        output_lines = []

        for target_number in range(1, prediction_window + 1):
            preds, matched_rule_ids, matched_supports = apply_rules_to_dataset(dataset.test,
                                                                               rules_df,
                                                                               target_number)

            dataset.test[f"PLP_target_{target_number}"] = preds
            dataset.test[f"PLP_rule_id_t+{target_number}"] = matched_rule_ids
            dataset.test[f"PLP_rule_support_t+{target_number}"] = matched_supports

            y_true = dataset.test[f"target_class_t+{target_number}"].astype(int)
            y_pred = dataset.test[f"PLP_target_{target_number}"].astype(int)

            pr = prediction_ratio(y_pred)
            acc = exact_accuracy(y_true, y_pred)
            acc1 = accuracy_at_one(y_true, y_pred)

            output_lines.append(f"Prediction Window {target_number}")
            output_lines.append(f"Prediction Ratio: {pr}")
            output_lines.append(f"Accuracy: {acc}")
            output_lines.append(f"Accuracy@1: {acc1}")
            output_lines.append("")

            logger.debug(f"PLP - Prediction Window {target_number}")
            logger.debug(f"Prediction Ratio: {pr}")
            logger.debug(f"Accuracy: {acc}")
            logger.debug(f"Accuracy@1: {acc1}")

        with open(model_dir_path + "/results.txt", "w") as f:
            f.write("\n".join(output_lines))

        dataset.test.to_csv(model_dir_path + "/PLP_preds.csv", index=False)

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)