AUTO-QA
# Setup
First setup virtual environment for development usign conda or pip

# Dataset Generation
## Step 1: Generate scene file for Image-Scenes

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


## Step 2: Generate Questions using scene file

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

# Baseline Models
## Step 1: Encoding question and Feature Extraction
 ### a) Encoding Question
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
## Step 2:Models
