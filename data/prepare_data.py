import os
if not os.path.exists("MLHomework_Toxicity/train_origin.csv"):
    if os.path.exists("MLHomework_Toxicity"):
        os.system("rm -rf MLHomework_Toxicity")
    if not os.path.exists("MLHomework_Toxicity.zip"):
        print("Downloading MLHomework_Toxicity.zip ...")
        os.system("wget https://drive.google.com/file/d/1nhh2G8pMLxqMxxHK2yzX_0lH-24pNAqp/view?usp=sharing")
    os.system("unzip MLHomework_Toxicity.zip")
    os.system("mv MLHomework_Toxicity/train.csv MLHomework_Toxicity/train_origin.csv")

import pandas as pd
import numpy as np
file_path = "MLHomework_Toxicity/train_origin.csv"
file = pd.read_csv(file_path)
df = pd.DataFrame(file)

eval_index = np.random.choice(range(df.shape[0]), int(df.shape[0] * 0.1), replace=False)
df_eval = df.loc[sorted(eval_index)]
df_eval.to_csv("MLHomework_Toxicity/eval.csv")

train_index = [i for i in range(df.shape[0]) if i not in eval_index]
df_train = df.loc[sorted(train_index)]
df_train.to_csv("MLHomework_Toxicity/train.csv")
