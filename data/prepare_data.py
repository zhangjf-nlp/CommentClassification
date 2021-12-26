import os
import json
import numpy as np
import pandas as pd

input_dir = "/home/zhangjf/CommentClassification/data/MLHomework_Toxicity"
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col_name in ['target'] + identity_columns:
        bool_df[col_name] = np.where(bool_df[col_name] >= 0.5, True, False)
    return bool_df

def subgroup_analysis(answers_extra, subgroup, intervals=1000, label_col='target'):
    diff = 1 / (intervals-1)
    x = np.linspace(0,1,intervals)
    interval_counts = [0]*intervals
    samples = answers_extra[subgroup].to_list()
    for sample in samples:
        interval_counts[int(sample/diff) if sample>=0 else 0] += 1
    pdf = np.array(interval_counts) / len(samples)
    cdf = np.cumsum(pdf)
    print(f"p({subgroup}<0.5) = {cdf[int(intervals/2)]}")
    
    answers = convert_dataframe_to_bool(answers_extra)
    
    cross_counts = np.array(
        [[sum(answers[subgroup] & answers[label_col]), sum(answers[subgroup] & ~answers[label_col])],
         [sum(~answers[subgroup] & answers[label_col]), sum(~answers[subgroup] & ~answers[label_col])]])
    print(f"{subgroup}-{label_col} counts:")
    print(cross_counts)
    
    cross_importance = np.zeros((2,2))
    counts_overall = cross_counts[0,0]+cross_counts[0,1]+cross_counts[1,0]+cross_counts[1,1]
    counts_subgroup = cross_counts[0,0]+cross_counts[0,1]
    counts_bnsp = cross_counts[0,0]+cross_counts[1,1]
    counts_bpsn = cross_counts[1,0]+cross_counts[0,1]
    
    cross_importance[0,0] = cross_counts[0,0] / counts_bnsp + \
                            cross_counts[0,0] / counts_subgroup + \
                            cross_counts[0,0] / counts_overall
    
    cross_importance[1,0] = cross_counts[1,0] / counts_bpsn + \
                            cross_counts[1,0] / counts_overall
    
    cross_importance[0,1] = cross_counts[0,1] / counts_bpsn + \
                            cross_counts[0,1] / counts_subgroup + \
                            cross_counts[0,1] / counts_overall
    
    cross_importance[1,1] = cross_counts[1,1] / counts_bnsp + \
                            cross_counts[1,1] / counts_overall
    
    cross_importance = cross_importance/4
    
    print(f"{subgroup}-{label_col} importance:")
    print(cross_importance)
    
    for indexer, importance, num_cases in zip(
        [answers[subgroup] & answers[label_col], answers[subgroup] & ~answers[label_col],
         ~answers[subgroup] & answers[label_col], ~answers[subgroup] & ~answers[label_col]],
        [cross_importance[0,0], cross_importance[0,1],
         cross_importance[1,0], cross_importance[1,1]], # the weight of such conditions
        [cross_counts[0,0], cross_counts[0,1],
         cross_counts[1,0], cross_counts[1,1]] # the number of cases of such conditions
    ):
        answers_extra.loc[indexer, 'weight'] += importance / num_cases # the sampling probability
    return

if __name__ == "__main__":
    if not os.path.exists("MLHomework_Toxicity/eval_extra_target_weight.csv"):
        if not os.path.exists("MLHomework_Toxicity/split_result.json"):
            if not os.path.exists("MLHomework_Toxicity/train_origin.csv"):
                if os.path.exists("MLHomework_Toxicity"):
                    os.system("rm -rf MLHomework_Toxicity")
                if not os.path.exists("MLHomework_Toxicity.zip"):
                    print("Stage.1 Downloading MLHomework_Toxicity.zip ...")
                    os.system("wget https://drive.google.com/file/d/1nhh2G8pMLxqMxxHK2yzX_0lH-24pNAqp/view?usp=sharing")
                
                print("Stage.2 Unziping MLHomework_Toxicity.zip ...")
                os.system("unzip MLHomework_Toxicity.zip")
                os.system("mv MLHomework_Toxicity/train.csv MLHomework_Toxicity/train_origin.csv")
                os.system("mv MLHomework_Toxicity/train_extra.csv MLHomework_Toxicity/train_origin_extra.csv")
            
            print("Stage.3 Splitting train-set and eval-set ...")
            df_origin = pd.read_csv("MLHomework_Toxicity/train_origin.csv")
            df_origin_extra = pd.read_csv("MLHomework_Toxicity/train_origin_extra.csv")
            
            num_samples = df_origin.shape[0]
            random_ordered_list = np.random.permutation(num_samples).tolist()
            eval_index = random_ordered_list[:int(num_samples/10)]
            train_index = random_ordered_list[int(num_samples/10):]

            df_origin.loc[sorted(eval_index)].to_csv("MLHomework_Toxicity/eval.csv")
            df_origin_extra.loc[sorted(eval_index)].to_csv("MLHomework_Toxicity/eval_extra.csv")

            df_origin.loc[sorted(train_index)].to_csv("MLHomework_Toxicity/train.csv")
            df_origin_extra.loc[sorted(train_index)].to_csv("MLHomework_Toxicity/train_extra.csv")

            with open("MLHomework_Toxicity/split_result.json", "w") as f:
                f.write(json.dumps({"train":train_index, "eval":eval_index}, ensure_ascii=False))
        else:
            with open("MLHomework_Toxicity/split_result.json", "r") as f:
                split_result = json.loads(f.read())
                train_index, eval_index = split_result["train"], split_result["eval"]
            df_origin = pd.read_csv("MLHomework_Toxicity/train_origin.csv")
            df_origin_extra = pd.read_csv("MLHomework_Toxicity/train_origin_extra.csv")

        print("Stage.4 Calculating sampling weights according the metric definition ...")
        df_origin_extra['target'] = df_origin['target']
        df_origin_extra['weight'] = 0

        import matplotlib.pyplot as plt

        for subgroup in identity_columns:
            subgroup_analysis(df_origin_extra, subgroup)
        
        df_origin_extra['weight'] = df_origin_extra['weight'] / len(identity_columns)
        print(f"answers_extra['weight'].sum(): {df_origin_extra['weight'].sum()}")
        df_origin_extra.to_csv("MLHomework_Toxicity/train_origin_extra_target_weight.csv")
        df_origin_extra.loc[sorted(eval_index)].to_csv("MLHomework_Toxicity/eval_extra_target_weight.csv")
        df_origin_extra.loc[sorted(train_index)].to_csv("MLHomework_Toxicity/train_extra_target_weight.csv")
    