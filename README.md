# AUTO-QA 
WELCOME to Autonomous Question Answering. This repo demonstrates how to reproduce results for AUTO-QA paper and also generate necessary question and scene files.
## Setup
### Install
It is recommend to use the Anaconda package manager. This code is tested on Ubuntu 16.04.
First setup virtual environment for development using conda (using following instructions)
```bash
   conda create --name <env_name> --file requirements.txt # or directly use yml file 
   conda env create -f environment.yml
   source ~/.bashrc
   conda activate [env_name]
```
Follow the steps given at [argoverse-api](https://github.com/argoai/argoverse-api) to install argoverse api
### Prepare Dataset
Download Training logs from [Argoverse](https://www.argoverse.org/data.html#download-link) site and create some directories as follows.Extract Training Part 1, 2, 3, 4 and Sample_dataset log file in `DATA/train/argoverse-tracking` and Validation log in `DATA/test/argoverse-tracking`

```plain
└── DATA <-Root dir for Dataset
       ├── train    <-- 66 training logs (from train 1,2,3,4 and sample_dataset in argoverse dataset)
       |   └── argoverse-tracking <-- contain all logs
       |       └── logs
       |
       └── test   25 test logs (from val in argoverse dataset)  
           └── argoverse-tracking
               └── logs
           
```

## Dataset Generation
### Step 1: Generate scene file for Image-Scenes

First we will create label file corresponding to each log file in lidar dataset by projectind 3D annotation into images.
```bash
   cd argo_preprocess
   python annotation_generate.py --input_dir='[path to dataset folder]'
```

Next we will create scene file, taking reference as the data collection vehicle.

```bash
   python scene_create.py --input_dir='[path to dataset folder]' --split='[train/test]'
```
The generated scene files for each log can be found in `output\[train\test]_scenes`


### Step 2: Generate Questions using scene file

```bash
   cd ../question_generation
   python collect_scenes.py --input_dir='../output/train_scenes.json' --output_file='../output/ARGO_train_scenes.json' --split='train'
   python collect_scenes.py --input_dir='../output/test_scenes.json' --output_file='../output/ARGO_test_scenes.json' --split='test'
```

`output/Argo_[train/test]_scenes` is the genreated collected scene file location.


```bash
   python generate_questions.py --input_scene_file='../output/ARGO_train_scenes.json' --output_questions_file='../output/ARGO_train_questions.json'
   python generate_questions.py --input_scene_file='../output/ARGO_test_scenes.json' --output_questions_file='../output/ARGO_test_questions.json'
```
Generate json file containing training and test set questions with answer and program at ```output/ARGO_[train/test]_questions.json```

## Baseline Models
### Step 1: Encoding question and Feature Extraction
 #### a) Encoding Question
```bash
   cd ../argo_preprocess
   python preprocess_questions.py  --input_questions_json='../output/ARGO_[train/test]_questions.json'  --output_h5_file='all_questions.h5' --output_vocab_json=' vocab_[train/test].json'
```
#### b) Train/Val Split

```bash
   python train_test_split.py   
```

#### c) Feature Extraction
```bash
   python extract_img_features.py --img_size=224  --root_dir='[PATH TO DATASET FOLDER]' --model_type='resnet101'
```
### Step 2: Models
   #### a) Image Based Models
   ```bash
      cd models
      export CUDA_VISIBLE_DEVICES=1
      python train.py 
      --model_type LSTM_BASIC   \
      --model_dir ../output/LSTM_BASIC \
      --image_features /data/ksumit/testing/output/processed/resnet101_features.h5 \
      --train_batch_size 2000 \
      --val_batch_size 1000 \
      --encoder_type lstm \
      --lr 1e-4 \
      --num_epochs 20 \
      --train_num_workers 4 \
      --val_num_workers 1 
   ```
   To add:
   
   1. ```--model_dir ``` directory to save checkpoint for each epoch
   2. ```--load_lidar ``` whether to load lidar data or not while dataloading
   3. ```--resume_training ``` to start training from saved checkpoint. In this must assign name of checkpoint to be loaded```--model_name=[checkpoint name]```, e.g. ```--model_name=SAN_gru_Ep29.pkl```.
   
   #### b) Point Cloud Based Models

   ```bash
      python  train.py \
         --model_type LIDAR_MODEL \
         --train_batch_size 20 \
         --model_dir ../output/LIDAR_MODEL \
         --val_batch_size 20 \
         --encoder_type gru \
         --lr 5e-4 \
         --num_epochs 20 \
         --train_num_workers 4 \
         --val_num_workers 1 
   ```
   #### c) Combination Models
   ```bash
      python  train_lidar.py \
         --model_type SAN_LIDAR \
         --train_batch_size 20 \
         --model_dir ../output/SAN_LIDAR \
         --val_batch_size 20 \
         --encoder_type gru \
         --lr 5e-4 \
         --num_epochs 20 \
         --train_num_workers 4 \
         --val_num_workers 1 
   ```

