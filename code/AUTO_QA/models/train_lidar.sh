export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1



############################################SAN########################################################

# python train_lidar.py \
#     --model_type SAN_LIDAR \
#     --train_batch_size 64 \
#     --model_dir ../output/SAN_LIDAR \
#     --image_features ../output/processed/vgg16_train_features.h5 \
#     --train_encodings ../output/processed/train_questions.h5 \
#     --val_encodings ../output/processed/val_questions.h5 \
#     --vocab ../output/processed/vocab_train.json \
#     --val_batch_size 64 \
#     --encoder_type gru \
#     --lr 5e-3 \
#     --num_epochs 40 \
#     --train_num_workers 4 \
#     --val_num_workers 2  \
#     | tee SAN_lidar.txt 

# for resuming training use 
#     --resume_training \
#     --model_name=SAN_LIDAR_gru_Ep38.pkl \


##############################################MFB#####################################
# python  train_lidar.py \
#     --model_type MFB_LIDAR \
#     --train_batch_size 32 \
#     --model_dir ../output/MFB_LIDAR \
#     --val_batch_size 32 \
#     --encoder_type gru \
#     --lr 1e-2 \
#     --num_epochs 20 \
#     --train_num_workers 2 \
#     --val_num_workers 2 \
#     --grouping single_scale \
#     --image_features ../output/processed/vgg16_train_features.h5 \
# 	  | tee MFB_lidar.txt 


##############################################MCB#######################################
# python train_lidar.py \
#     --model_type MCB_LIDAR \
#     --train_batch_size 20 \
#     --model_dir ../output/MCB_LIDAR \
#     --image_features ../output/processed/vgg16_train_features.h5 \
#     --val_batch_size 20 \
#     --encoder_type gru \
#     --lr 5e-3 \
#     --num_epochs 40 \
#     --grouping single_scale \
#     --train_num_workers 2 \
#     --val_num_workers 2  \
#     | tee MCB_lidar.txt 

############################################MLB########################################################

# python train_lidar.py \
#     --model_type MLB_LIDAR \
#     --train_batch_size 64 \
#     --model_dir ../output/MLB_LIDAR \
#     --image_features ../output/processed/vgg16_train_features.h5 \
#     --val_batch_size 64 \
#     --encoder_type lstm \
#     --lr 5e-3 \
#     --num_epochs 21 \
#     --grouping single_scale \
#     --train_num_workers 2 \
#     --val_num_workers 2  \
#     | tee MLB_lidar.txt 

############################################MUTAN########################################################

python train_lidar.py \
    --model_type MUTAN_LIDAR \
    --train_batch_size 20 \
    --model_dir ../output/MUTAN_LIDAR \
    --image_features ../output/processed/vgg16_features.h5 \
    --train_encodings ../output/processed/train_questions.h5 \
    --val_encodings ../output/processed/val_questions.h5 \
    --vocab ../output/processed/vocab_train.json \
    --val_batch_size 20 \
    --encoder_type lstm \
    --grouping multi_scale \
    --lr 5e-3 \
    --num_epochs 30 \
    --train_num_workers 4 \
    --val_num_workers 2  \
    | tee MUTAN_lidar.txt 
