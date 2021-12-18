# CommentClassification
The homework of ML-2021-BUAA. A project for multi-label text label regression.

## Step 0. setup environment
```
pip install requirements.txt
```

## Step 1. prepare data
```
cd data
python3 prepare_data.py
cd ..
```

## Step 2. run experiment
```
python3 experiment.py [--head_class BasicRegreesionHead] [--loss_class mse] [--extra_counts 6]
```

## Step 3. predict
```
python3 experiment.py [--head_class BasicRegreesionHead] [--loss_class mse] [--extra_counts 6] --test
```
