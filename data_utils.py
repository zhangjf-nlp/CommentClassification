import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_LENGTH = 200

def create_dataloader(root="./data/MLHomework_Toxicity", usage="train", tokenizer=None, batch_size=32):
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    file_path = f"{root}/{usage}.csv"
    file = pd.read_csv(file_path)
    df = pd.DataFrame(file)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([tokenizer.encode(text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
                      for text in tqdm(df["comment_text"])]).long(),
        torch.tensor([_ for _ in df["target"]] if usage=="train" else [np.nan for i in range(df.shape[0])]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        sampler = torch.utils.data.RandomSampler(dataset) if usage=="train" else torch.utils.data.SequentialSampler(dataset),
        batch_size = batch_size,
        drop_last = True if usage=="train" else False
    )
    return dataloader

def get_dataloader(root="./data/MLHomework_Toxicity", usage="train", tokenizer=None, batch_size=32, erase=False):
    dataloader_file_path = f"{root}/{usage}-{batch_size}.pt"
    if os.path.exists(dataloader_file_path) and not erase:
        dataloader = torch.load(dataloader_file_path)
    else:
        dataloader = create_dataloader(root, usage, tokenizer, batch_size)
        torch.save(dataloader, dataloader_file_path)
    return dataloader