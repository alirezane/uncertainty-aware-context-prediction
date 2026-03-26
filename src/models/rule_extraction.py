import os
import logging
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

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

max_depth = 6
min_samples_split = 20
min_samples_leaf = 10
criterion = "gini"
random_state = 42
target_number = 1


def tree_to_rules_text(tree_model, feature_names):
    return export_text(tree_model, feature_names=list(feature_names))


def extract_leaf_rules(tree_model, feature_names, class_names=None):
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1 = list(path)
            p1.append(f"({name} <= {threshold})")
            recurse(tree_.children_left[node], p1, paths)
            p2 = list(path)
            p2.append(f"({name} > {threshold})")
            recurse(tree_.children_right[node], p2, paths)
        else:
            values = tree_.value[node][0]
            class_id = int(values.argmax())
            sample_count = int(values.sum())
            if class_names is not None and class_id < len(class_names):
                pred = class_names[class_id]
            else:
                pred = class_id
            paths.append((path, pred, sample_count))

    recurse(0, path, paths)
    return paths


if __name__ == '__main__':
    try:
        model_dir_path = util_classification.make_model_dir("./models/neuro_symbolic", "rule_extraction")

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

        X_train = dataset.train_X
        y_train = dataset.train_y[:, target_number - 1]

        logger.debug(f"Training rule extraction tree")
        logger.debug(f"target_number = {target_number}")
        logger.debug(f"max_depth = {max_depth}")
        logger.debug(f"min_samples_split = {min_samples_split}")
        logger.debug(f"min_samples_leaf = {min_samples_leaf}")
        logger.debug(f"criterion = {criterion}")

        clf = DecisionTreeClassifier(max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     criterion=criterion,
                                     random_state=random_state)

        clf.fit(X_train, y_train)

        feature_names = dataset.feature_columns
        class_names = [0, 1, 2, 3, 4]

        rules_text = tree_to_rules_text(clf, feature_names)
        with open(model_dir_path + f"/decision_tree_rules_target_{target_number}.txt", "w") as f:
            f.write(rules_text)

        leaf_rules = extract_leaf_rules(clf, feature_names, class_names)

        rules_rows = []
        for i in range(len(leaf_rules)):
            conditions, pred, sample_count = leaf_rules[i]
            if len(conditions) == 0:
                rule_body = "true"
            else:
                rule_body = " and ".join(conditions)
            rules_rows.append({
                "rule_id": i + 1,
                "target_number": target_number,
                "predicted_class": pred,
                "sample_count": sample_count,
                "rule": rule_body
            })

        rules_df = pd.DataFrame(rules_rows)
        rules_df.to_csv(model_dir_path + f"/decision_tree_rules_target_{target_number}.csv", index=False)

        plp_lines = []
        for _, row in rules_df.iterrows():
            rule_body = row["rule"]
            pred = row["predicted_class"]
            sample_count = row["sample_count"]
            plp_lines.append(f"occupancy_class_t{target_number}({pred}) :- {rule_body}.")
            plp_lines.append(f"rule_support({int(row['rule_id'])},{sample_count}).")

        with open(model_dir_path + f"/decision_tree_rules_target_{target_number}_plp.txt", "w") as f:
            f.write("\n".join(plp_lines))

        logger.debug(f"rules extracted: {rules_df.shape[0]}")

    except BaseException as e:
        logger.exception('An exception was thrown!', exc_info=True)