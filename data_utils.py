import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler

columns_to_pearsons = [
    ('insult', 0.928228709),
    ('obscene', 0.49313278),
    ('identity_attack', 0.449813009),
    ('severe_toxicity', 0.393436345),
    ('threat', 0.287890162),
    ('sexual_explicit', 0.252612402),
    ('toxicity_annotator_count', 0.236762562),
    ('white', 0.193880459),
    ('black', 0.166748936),
    ('muslim', 0.134063242),
    ('homosexual_gay_or_lesbian', 0.130827878),
    ('male', 0.073764717),
    ('female', 0.063034719),
    ('psychiatric_or_mental_illness', 0.055567062),
    ('jewish', 0.047304922),
    ('transgender', 0.042134754),
    ('heterosexual', 0.038090143),
    ('other_race_or_ethnicity', 0.03603415),
    ('intellectual_or_learning_disability', 0.035986512),
    ('other_sexual_orientation', 0.032802388),
    ('latino', 0.02894989),
    ('identity_annotator_count', 0.025081523),
    ('disagree', 0.024462097),
    ('publication_id', 0.021702565),
    ('bisexual', 0.021267946),
    ('other_religion', 0.020937682),
    ('likes', 0.019121949),
    ('sad', 0.018002533),
    ('other_gender', 0.012471848),
    ('wow', 0.012345404),
    ('atheist', 0.010485192),
    ('other_disability', 0.007868749),
    ('physical_disability', 0.007732546),
    ('article_id', 0.007709933),
    ('asian', 0.006988363),
    ('buddhist', 0.004597721),
    ('hindu', 0.003781904),
    ('parent_id', -0.002296104),
    ('christian', -0.006758356)
]

def get_df(root="./data/MLHomework_Toxicity", usage="train"):
    return pd.DataFrame(pd.read_csv(f"{root}/{usage}.csv"))
    
def create_dataloader(args, root="./data/MLHomework_Toxicity", usage="train", tokenizer=None, extra_counts=6, erase=False):
    dataset_file_path = f"{root}/{args.pretrained_model_name_or_path}/{'bias-' if args.bias_sampling else ''}{usage}-{args.max_length}-(1+{extra_counts}).pt"
    if os.path.exists(dataset_file_path) and not erase:
        dataset = torch.load(dataset_file_path)
    else:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
        df = get_df(root, usage)
        extra_columns = [a for a,b in columns_to_pearsons[:extra_counts]]
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([tokenizer.encode(text, truncation=True, max_length=args.max_length, padding="max_length")
                          for text in tqdm(df["comment_text"])]).long(),
            torch.tensor([_ for _ in df["target"]] if usage in ["train","eval"] else np.zeros(df.shape[0])).float(),
            torch.tensor(get_df(root, f"{usage}_extra")[extra_columns].to_numpy() if usage in ["train","eval"] else np.zeros([df.shape[0], len(extra_columns)])).float(),
        )
        if not os.path.exists(os.path.dirname(dataset_file_path)):
            os.makedirs(os.path.dirname(dataset_file_path))
        torch.save(dataset, dataset_file_path)
    
    if args.bias_sampling and usage in ["train","eval"]:
        weights = get_df(root, f"{usage}_extra_target_weight")["weight"].to_list()
        assert len(weights)==len(dataset)
        sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = RandomSampler(dataset) if usage=="train" else SequentialSampler(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        sampler = sampler,
        batch_size = args.batch_size,
        drop_last = True if usage=="train" else False
    )
    return dataloader

def get_dataloader(args, root="./data/MLHomework_Toxicity", usage="train", tokenizer=None, extra_counts=6, erase=False):
    dataloader_file_path = f"{root}/{args.pretrained_model_name_or_path}/{'bias-' if args.bias_sampling else ''}{usage}-{args.max_length}-{args.batch_size}*(1+{extra_counts}).pt"
    if os.path.exists(dataloader_file_path) and not erase:
        dataloader = torch.load(dataloader_file_path)
    else:
        dataloader = create_dataloader(args, root, usage, tokenizer)
        if not os.path.exists(os.path.dirname(dataloader_file_path)):
            os.makedirs(os.path.dirname(dataloader_file_path))
        torch.save(dataloader, dataloader_file_path)
    return dataloader

def export_result(args, root="./data/MLHomework_Toxicity", usage="eval"):
    df, df_extra = get_df(usage=usage), get_df(usage="train_extra")
    df_extra = df_extra.loc[list(df['Unnamed: 0'])]
    all_target, all_pred = torch.load(f"{args.exp_dir}/target_pred.pt")
    df_extra["target"] = all_target
    df_extra["pred"] = all_pred
    df_extra.to_csv(f"{args.exp_dir}/res_{usage}.csv")