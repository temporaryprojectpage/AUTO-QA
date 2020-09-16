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
This codebase use argoverse-api. Follow the steps given at [argoverse-api](https://github.com/argoai/argoverse-api) to install argoverse api
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
   python extract_img_features.py --img_size=224  --root_dir='[PATH TO DATASET FOLDER]' --model_type='resnet152'  #for simple CNN_LSTM MODEL(2048 dim)
   python extract_img_features.py --img_size=448  --root_dir='[PATH TO DATASET FOLDER]' --model_type='vgg16'      #for all other models(512x14x14 dim)
```
### Step 2: Models
   **It is  recommended to use script files ```train.sh, train_lidar.sh, inference.sh, visualize.sh```  for prefilled arguments.**
   #### a) Image Based Models
   ```bash
      cd models
      export CUDA_VISIBLE_DEVICES=1 #1,2,3 for multi-gpu training
      python  train.py \
	    --model_type SAN \
	    --train_batch_size 300 \
	    --model_dir ../output/SAN \
	    --val_batch_size 100 \
	    --encoder_type gru \
	    --lr 5e-2 \
	    --fusion_type concat \
	    --num_epochs 30 \
	    --train_num_workers 4 \
	    --val_num_workers 2 \
	    --image_features ../output/processed/vgg16_train_features.h5 \
	    --train_encodings ../output/processed/train_questions.h5 \
	    --val_encodings ../output/processed/val_questions.h5 \
	    --vocab ../output/processed/vocab_train.json \
	    | tee san.txt
   ```
   
   #### b) Point Cloud Based Models

   ```bash
      python  train.py \
	    --model_type LIDAR_MODEL \
	    --train_batch_size 10 \
	    --model_dir ../output/LIDAR_MODEL \
	    --val_batch_size 10 \
	    --encoder_type gru \
	    --load_lidar \
	    --lr 5e-4 \
	    --grouping multi_scale \
	    --num_epochs 20 \
	    --train_num_workers 4 \
	    --val_num_workers 1 \
	    --image_features ../output/processed/vgg16_train_features.h5 \
	    --train_encodings ../output/processed/train_questions.h5 \
	    --val_encodings../output/processed/val_questions.h5 \
	    --vocab ../output/processed/vocab_train.json \
	    | tee lidar_model.txt 
   ```
   To add:
   
   1. ```--model_dir ``` directory to save checkpoint for each epoch
   2. ```--load_lidar ``` whether to load lidar data or not while dataloading
   3. ```--resume_training ``` to start training from saved checkpoint. In this must assign name of checkpoint to be loaded ```--model_name=[checkpoint name]```,    e.g. ```--model_name=SAN_gru_Ep29.pkl```.
   4. ```--fusion_type={'concat','dot'}``` for `CNN_LSTM` and `LIDAR_MODEL` model and ```--fusion_type={'concat','hierarchical'}``` for simple concat model or hierarchical which use our level-2 attention for all other models
   5. ```--model_type={'LSTM_BASIC','CNN_LSTM','SAN','MCB','DAN','MFB','LIDAR_MODEL','MUTAN','MLB'}``` for chosing different level-1 attention
   6. ```--grouping={'single_scale','multi_scale'}``` for different backbone network for Point Cloud Based Model (`LIDAR_MODEL`)
   7. ```--image_features={resnet152_features,vgg16_features}``` resnet152 features for `CNN_LSTM` and all other image based model use vgg16 features.
   
   #### c) Combination Models
   ```bash
      python train_lidar.py \
            --model_type MUTAN_LIDAR \
            --train_batch_size 20 \
            --model_dir ../output/MUTAN_LIDAR \
            --image_features ../output/processed/vgg16_train_features.h5 \
            --train_encodings ../output/processed/train_questions.h5 \
            --val_encodings ../output/processed/val_questions.h5 \
            --vocab ../output/processed/vocab_train.json \
            --val_batch_size 20 \
            --encoder_type lstm \
            --grouping multi_scale \
            --lr 5e-3 \
            --num_epochs 30 \
            --grouping single_scale \
            --train_num_workers 4 \
            --val_num_workers 2  \
            |  tee MUTAN_lidar.txt 
   ```
   
   #### d) Visualization
   ```bash
      cd visualization
      python  visualize.py \
	    --model_type SAN \
	    --model_dir ../output/SAN_vgg16_sigmoid_new_quesition \
	    --model_name=SAN_gru_Ep29.pkl \
	    --save_dir=../../output/results_images_SAN \
	    --val_batch_size 100 \
	    --encoder_type gru \
	    --val_num_workers 1 \
	    --image_features ../output/processed/vgg16_train_features.h5 \
	    --val_encodings ../output/processed/val_questions.h5 \
	    --vocab ../output/processed/vocab_train.json \
	    | tee log_san_images.txt 
   ```
   To add:
   
   1. ```--save_dir ``` directory to save images for after attention visualization
   2.```--model_name=[checkpoint name]``` to load checkpoint for attention visualization, e.g. ```--model_name=SAN_gru_Ep29.pkl```.
   
   #### e) Inference
   ```bash
      python  inference.py \
	    --model_type MUTAN \
	    --model_dir ../output/MUTAN \
	    --model_name=MUTAN_lstm_Ep28.pkl \
	    --test_batch_size 100 \
	    --encoder_type lstm \
	    --test_num_workers 1 \
	    --image_features ../output/processed/vgg16_test_features.h5 \
	    --fusion_type hierarchical \
	    --test_encodings ../output/processed/test_questions.h5 \
	    --vocab ../output/processed/vocab_test.json \
	    | tee test_mutan.txt
   ```
   
#### References: 
For question generation and preprocessing we have used [CLEVR](https://github.com/facebookresearch/clevr-iep), [CLEVR-IEP](https://github.com/facebookresearch/clevr-dataset-gen).<br />
For testing different existing vqa models and pointnet++ implementation , we have refered [MLB, MUTAN, MCB](https://github.com/Cadene/vqa.pytorch), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), [Dan](https://github.com/tzuhsial/pytorch-vqa-dan), [SAN](https://github.com/rshivansh/San-Pytorch), [MFB](https://github.com/yuzcccc/vqa-mfb).

