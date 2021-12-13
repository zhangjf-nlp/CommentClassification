from cuda_utils import dynamic_cuda_allocation
from cuda_utils import block_until_cuda_memory_free
dynamic_cuda_allocation()
block_until_cuda_memory_free(required_mem=1000)

import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import logging
logging.set_verbosity_error()

from data_utils import get_dataloader

def init_config(args_specification=None):
    parser = argparse.ArgumentParser(description="Comment Classification study")
    
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--tiny_experiment', action='store_true', default=False, help="only use a tiny subset to train/eval/test")
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='bert-base-uncased', help="the specified path to load pretrained vae")
    
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=100, help="the size of mini-batch in training")
    
    parser.add_argument('--seed', type=int, default=123456, help="random seed")
    parser.add_argument('--opt', type=str, choices=["adamw"], default="adamw", help="optimizer")
    parser.add_argument('--scheduler_style', type=str, choices=["dynamic", "static"], default="dynamic", help="scheduler")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight deay if we apply some.")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help="Epsilon for Adam optimizer.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--head_class', type=str, default="BasicRegressionHead")
    
    if args_specification:
        args = parser.parse_args([_ for _ in args_specification if _])
    else:
        args = parser.parse_args()
    
    assert torch.cuda.is_available(), f"this project only supports gpu at the moment while torch.cuda.is_available() is False"
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    from modules.baseline import available_head_classes
    assert args.head_class in available_head_classes, f"unknown head_class: {args.head_class}, available: {[_ for _ in available_head_classes.keys()]}"
    args.head_class = available_head_classes[args.head_class]
    
    args.model_name = f"BERT_with_{args.head_class.__name__}"
    
    if args.exp_dir == None:
        args.exp_dir = f"exp/{args.model_name}"
    
    args.logging = create_logging(args, erase=True)
    args.logging(args)
    
    return args

def create_logging(args, erase=True):
    import functools, shutil
    def logging(output, log_path, end="\n", do_print=True):
        if do_print:
            print(output)
        if log_path is not None:
            with open(log_path, "a+") as logf:
                logf.write(str(output) + end)
    
    if args.test:
        return functools.partial(logging, log_path=None, do_print=False)
    
    if os.path.exists(args.exp_dir):
        if erase:
            print(f"Path {args.exp_dir} exists. Remove and remake.")
            shutil.rmtree(args.exp_dir)
            os.makedirs(f"{args.exp_dir}/chkpts")
            os.makedirs(f"{args.exp_dir}/runs")
            return functools.partial(logging, log_path=f"{args.exp_dir}/log.txt")
        else:
            print(f"Path {args.exp_dir} exists. Log after the existing content.")
            return functools.partial(logging, log_path=f"{args.exp_dir}/log.txt")
    
    print(f"Create new experiment directory: {args.exp_dir}.")
    os.makedirs(f"{args.exp_dir}/chkpts")
    os.makedirs(f"{args.exp_dir}/runs")
    return functools.partial(logging, log_path=f"{args.exp_dir}/log.txt")

def create_model(args):
    from modules.baseline import Model
    model = Model(args)
    args.logging(model)
    return model.cuda()

def create_dataloader(args):
    train_dataloader = get_dataloader(usage="train", batch_size=args.batch_size)
    eval_dataloader = get_dataloader(usage="eval", batch_size=args.batch_size)
    args.train_steps = len(train_dataloader) * args.epoch
    args.warmup_steps = int(args.train_steps / 10)
    return train_dataloader, eval_dataloader
    
def create_optimizer_and_scheduler(args, model):
    from transformers import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    parameters_to_optimize = [n for n in model.named_parameters()]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters_to_optimize if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler_style == "dynamic":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=1)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps)
    
    args.best_eval_loss = 1e10
    return optimizer, scheduler

def create_tbwriter(args, erase=True):
    for path in [f"{args.exp_dir}/runs/train", f"{args.exp_dir}/runs/eval"]:
        if erase and os.path.exists(path):
            import shutil
            shutil.rmtree(path)
    from tensorboardX import SummaryWriter
    train_tbwriter = SummaryWriter(f"{args.exp_dir}/runs/train")
    eval_tbwriter = SummaryWriter(f"{args.exp_dir}/runs/eval")
    args.global_step = 0
    return train_tbwriter, eval_tbwriter

def train_epoch(args, model, train_dataloader, eval_dataloader, train_tbwriter, eval_tbwriter, optimizer, scheduler):
    model.train()
    dataloader, tbwriter = train_dataloader, train_tbwriter
    steps_per_epoch = len(dataloader) * 0.1 if args.tiny_experiment else len(dataloader)
    training_logging_steps = int(steps_per_epoch / 20)
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data_batch in bar:
        text, target = data_batch
        text, target = text.cuda(), target.cuda()
        loss, pred = model(text, target)
        bar.set_description(f"loss={loss.item():.2f}")
        for name, value in [
            ("loss", loss.item()),
        ]:
            tbwriter.add_scalar(name, value, args.global_step)
        
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        args.global_step += 1
        
        if args.global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            if not args.scheduler_style == "dynamic":
                scheduler.step()
            optimizer.zero_grad()
        
        if args.global_step % training_logging_steps == 0:
            args.logging(f"{str(bar)}", do_print=False)
            eval_epoch(args, model, eval_dataloader, eval_tbwriter, scheduler)
        
        if args.tiny_experiment and step > len(dataloader)*0.1:
            break
    return

@torch.no_grad()
def eval_epoch(args, model, dataloader, tbwriter, scheduler):
    model.eval()
    
    total_loss, total_samples = 0, 0
    bar = enumerate(dataloader)
    for step, data_batch in bar:
        text, target = data_batch
        text, target = text.cuda(), target.cuda()
        loss, pred = model(text, target)
        total_loss += loss
        total_samples += text.shape[0]
        
        if args.tiny_experiment and step > len(dataloader)*0.1:
            break
    
    eval_loss = total_loss / total_samples
    args.logging(f"Eval -- {args.global_step}: ")
    for name, value in [
        ("loss", eval_loss),
    ]:
        tbwriter.add_scalar(name, value, args.global_step)
        args.logging(f"{name} = {value}")
    
    if args.scheduler_style == "dynamic":
        scheduler.step(eval_loss)
    
    if eval_loss < args.best_eval_loss:
        args.best_eval_loss = eval_loss
        args.best_state_dict = model.state_dict()
        args.logging(f"update best eval loss to: {eval_loss:.3f}\n")
        
    model.train()

if __name__ == "__main__":
    
    args = init_config()
    
    try:
        args.logging(f"This experiemnt started at: {time.ctime()}")
        model = create_model(args)
        train_dataloader, eval_dataloader = create_dataloader(args)
        optimizer, scheduler = create_optimizer_and_scheduler(args, model)
        train_tbwriter, eval_tbwriter = create_tbwriter(args)
        args.logging(f"begin training")
        args.best_state_dict = None
        for epoch in range(args.epoch):
            args.logging(f"epoch {epoch+1}/{args.epoch}")
            train_epoch(args, model, train_dataloader, eval_dataloader, train_tbwriter, eval_tbwriter, optimizer, scheduler)
        
        args.logging(f"finish training: best_loss = {args.best_eval_loss}\n")
        
        if args.best_state_dict is not None:
            torch.save(args.best_state_dict, f"{args.exp_dir}/chkpts.pt")
            model.load_state_dict(args.best_state_dict)
        elif os.path.exists(f"{args.exp_dir}/chkpts/{args.stage}.pt"):
            model.load_state_dict(torch.load(f"{args.exp_dir}/chkpts.pt"))
        
        # TODO test
        
    except Exception as e:
        import traceback
        args.logging(f'traceback.format_exc():\n{traceback.format_exc()}')