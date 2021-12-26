# CommentClassification
The homework of ML-2021-BUAA. A project for multi-label text label regression.

## Step 0. setup environment
```
pip install -r requirements.txt
```

## Step 1. prepare data
```
cd data
python3 prepare_data.py
cd ..
```
if your machine does not have access to Google Drive, you can also download the dataset file through:

bhpan.buaa.edu.cn:
https://bhpan.buaa.edu.cn:443/link/F36CE716135C22426F1BA945C2789F30

pan.baidu.com:
https://pan.baidu.com/s/1ytJH042t5ALM79lGcG4NdQ token=0ygv

and save it as ./data/MLHomework_Toxicity.zip before the commands above

## Step 2. run experiment
```
python3 experiment.py --head TwoLayerRegressionHead --loss ce ----bias_sampling --pretrain microsoft/deberta-v3-large --batch_size 8 --gradient 4
```
following are more details for hyper-parameters settings:
```
usage: experiment.py [-h] [--test] [--bias_sampling] [--tiny_experiment]
                     [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                     [--epoch EPOCH] [--learning_rate LEARNING_RATE]
                     [--batch_size BATCH_SIZE] [--max_length MAX_LENGTH]
                     [--seed SEED] [--opt {adamw}]
                     [--scheduler_style {dynamic,static}]
                     [--weight_decay WEIGHT_DECAY]
                     [--adam_epsilon ADAM_EPSILON]
                     [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                     [--exp_dir EXP_DIR] [--freeze_pretrained]
                     [--agg_class {Aggregator,DebertaAggregator}]
                     [--head_class {BasicRegressionHead,TwoLayerRegressionHead,SimCLR_MLP,DebertaRegressionHead,ReluRegressionHead}]
                     [--loss_class {mse,ce}] [--extra_counts EXTRA_COUNTS]

Comment Classification study

optional arguments:
  -h, --help            show this help message and exit
  --test
  --bias_sampling       use Bias-AUC oriented sampling strategy
  --tiny_experiment     only use a tiny subset to train/eval/test
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        the specified path to load pretrained LM
  --epoch EPOCH
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
                        the size of mini-batch in training
  --max_length MAX_LENGTH
                        the max length of text tokenization
  --seed SEED           random seed
  --opt {adamw}         optimizer
  --scheduler_style {dynamic,static}
                        scheduler
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --adam_epsilon ADAM_EPSILON
                        Epsilon for Adam optimizer.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
  --exp_dir EXP_DIR
  --freeze_pretrained   freeze the pretrained weights
  --agg_class {Aggregator,DebertaAggregator}
  --head_class {BasicRegressionHead,TwoLayerRegressionHead,SimCLR_MLP,DebertaRegressionHead,ReluRegressionHead}
  --loss_class {mse,ce}
  --extra_counts EXTRA_COUNTS
                        number of extra features for joint learning
```
## Step 3. registry and submit
modify data/MLHomeworks_client/group_info.py to your settings, and run the following commands
```
python3 data/MLHomeworks_client/registry.py # input "Y" for twice to confirm registration
python3 data/MLHomeworks_client/client.py ./ # this command recursively posts all the submission.txt under this directory
```
for more functions (e.g. ensemble submissions), please refer to data/MLHomeworks_client/client.py