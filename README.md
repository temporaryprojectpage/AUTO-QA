# AUTO-QA

## Step 1: Generate scene file for Image-Scenes

First we will create label file corresponding to each log file in lidar dataset by projectind 3D annotation into images.
```bash
   cd argo_preprocess
   python annotation_generate.py --input_dir=[path to dataset folder]
```

Next we will create scene file, taking reference as the data collection vehicle.

```bash
   python scene_create.py --input_dir=[path to dataset folder] --split=[train/test]
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
   python generate_questions.py '--input_scene_file'='../output/ARGO_train_scenes.json' '--output_questions_file'=../output/ARGO_train_questions.json'
   python generate_questions.py '--input_scene_file'='../output/ARGO_test_scenes.json' '--output_questions_file'=../output/ARGO_test_questions.json'
```
Generate json file containing training and test set questions with answer and program at ```output/ARGO_[train/test]_questions.json```


## Step 3: Encoding question and Feature Extraction


