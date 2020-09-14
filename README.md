# AUTO-QA

## Step 1: Generate Scene file for Image-Scnes

First we will create label file corresponding to each log file in lidar dataset by projectind 3D annotation into images.
```bash
   cd argo_preprocess
   python annotation_generate.py --input_dir <path to dataset folder>
```

Next we will create scene file, taking reference as the data collection vehicle.

```bash
   python scene_create.py --input_dir <path to dataset folder>
```


