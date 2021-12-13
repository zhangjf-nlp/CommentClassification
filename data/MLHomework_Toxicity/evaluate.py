#!/usr/bin/env python
import sys, os, os.path
import os
import pandas as pd
import numpy as np

from sklearn import metrics

input_dir = sys.argv[1]
output_dir = sys.argv[2]
submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')
#######################
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'


def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])


def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = dict(
            subgroup=subgroup,
            subgroup_size=len(dataset[dataset[subgroup]])
        )
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


def evaluate(data_with_pred, model_name='pred'):
    bias_metrics_df = compute_bias_metrics_for_model(data_with_pred,
                                                     identity_columns,
                                                     model_name,
                                                     TOXICITY_COLUMN)

    Bias_AUC = get_final_metric(bias_metrics_df, calculate_overall_auc(data_with_pred,
                                                                       model_name))
    return Bias_AUC


def file_2_dict(f_pth):
    results = dict()
    if os.path.exists(f_pth):
        with open(f_pth, 'r') as f:
            lines = list(f.readlines())
        for l in lines:
            l = l.strip()
            test_id, toxicity = l.split(' ')
            results[int(test_id)] = float(toxicity)
    else:
        raise Exception('%s file does not exist' % f_pth)
    return results


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)


def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df


def read_csv(f_pth):
    data = pd.read_csv(f_pth)
    return data


if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')

    gold_list = os.listdir(truth_dir)
    for gold in gold_list:
        gold_file = os.path.join(truth_dir, gold)
        corresponding_submission_file = os.path.join(submit_dir, 'submission.txt')
        results = file_2_dict(corresponding_submission_file)
        answers = read_csv(gold_file)
        answers = convert_dataframe_to_bool(answers)

        pred = np.zeros(len(answers['id']))
        for i, test_id in enumerate(answers['id'].tolist()):
            if test_id in results.keys():
                pred[i] = results[test_id]
            else:
                pred[i] = 0.0
        answers['pred'] = pred
        Bias_AUC = evaluate(answers, 'pred')
        print(Bias_AUC)
        output_file.write("Bias_AUC: %lf" % Bias_AUC)
    output_file.close()
