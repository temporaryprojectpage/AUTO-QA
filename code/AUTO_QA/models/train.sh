export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2


###################LSTM########################################################
# python train.py  \
#     --model_type LSTM_BASIC   \
#     --model_dir ../output/LSTM_BASIC \
#     --image_features ../output/processed/resnet152_train_features.h5 \
#     --train_batch_size 2000 \
#     --val_batch_size 1000 \
#     --encoder_type lstm \
#     --lr 1e-4 \
#     --num_epochs 20 \
#     --train_num_workers 1 \
#     --val_num_workers 1 \
#     --train_encodings ../output/processed/train_questions.h5 \
#     --val_encodings ../output/processed/val_questions.h5 \
#     --vocab ../output/processed/vocab_train.json \
#     | tee lstm_log.txt

###################################CNN_LSTM############################################

# python train.py \
#     --model_type CNN_LSTM  \
#     --train_batch_size 300 \
#     --model_dir ../output/CNN_LSTM \
#     --val_batch_size 50 \
#     --encoder_type lstm \
#     --lr 5e-4 \
#     --fusion_type concat \ #concat/hierarchical 
#     --num_epochs 10 \
#     --train_num_workers 4 \
#     --val_num_workers 1  \
#     --image_features ../output/processed/resnet152_train_features.h5 \
#     | tee cnn_lstm.txt 


##############################################SAN#######################################

# python  train.py \
#     --model_type SAN \
#     --train_batch_size 300 \
#     --model_dir ../output/SAN \
#     --val_batch_size 100 \
#     --encoder_type gru \
#     --lr 5e-2 \
#     --fusion_type concat \
#     --num_epochs 30 \
#     --train_num_workers 4 \
#     --val_num_workers 2 \
#     --image_features ../output/processed/vgg16_train_features.h5 \
#     --train_encodings ../output/processed/train_questions.h5 \
#     --val_encodings ../output/processed/val_questions.h5 \
#     --vocab ../output/processed/vocab_train.json \
#     | tee san.txt

    #  --resume_training \
    #  --model_name=     \

##############################################MCB#######################################
# python  train.py \
#     --model_type MCB \
#     --train_batch_size 20 \
#     --model_dir ../output/MCB \
#     --val_batch_size 20 \
#     --encoder_type gru \
#     --lr 5e-4 \
#     --fusion_type concat \ #concat/hierarchical
#     --num_epochs 20 \
#     --train_num_workers 4 \
#     --val_num_workers 1 \
#     --image_features ../output/processed/vgg16_train_features.h5 \
# 	  | tee mcb.txt 


##############################################DAN#######################################
# python  train.py \
#     --model_type DAN \
#     --train_batch_size 128 \
#     --model_dir ../output/DAN \
#     --val_batch_size 128 \
#     --encoder_type gru \
#     --lr 1e-3 \
#     --fusion_type concat \ #concat/hierarchical
#     --num_epochs 30 \
#     --train_num_workers 4 \
#     --val_num_workers 2 \
#     --image_features ../output/processed/vgg16_train_features.h5 \
# 	| tee dan_log_vgg_concat.txt 


##############################################MFB##F#####################################
# python  train.py \
#     --model_type MFB \
#     --train_batch_size 128 \
#     --model_dir ../output/MFB \
#     --val_batch_size 128 \
#     --encoder_type gru \
#     --lr 1e-3 \
#     --num_epochs 20 \
#     --fusion_type concat \ #concat/hierarchical
#     --train_num_workers 4 \
#     --val_num_workers 1 \
#     --train_encodings train_questions2.h5 \
#     --val_encodings val_questions2.h5 \
#     --vocab vocab2.json \
#     --image_features ../output/processed/vgg16_train_features.h5 \
# 	  | tee MFB.txt 




############################################MLB########################################################

# python train.py \
#     --model_type MLB \
#     --train_batch_size 256 \
#     --model_dir ../output/MLB \
#     --image_features vgg16_train_features.h5 \
#     --val_batch_size 256 \
#     --encoder_type lstm \
#     --lr 5e-3 \
#     --fusion_type concat \ #concat/hierarchical
#     --num_epochs 21 \
#     --train_num_workers 4 \
#     --val_num_workers 2  \
#     | tee MLB.txt 


############################################MUTAN########################################################

# python train.py \
#     --model_type MUTAN \
#     --train_batch_size 128 \
#     --model_dir ../output/MUTAN \
#     --image_features vgg16_train_features.h5 \
#     --val_batch_size 128 \
#     --encoder_type lstm \
#     --lr 5e-3 \
#     --fusion_type concat \ #concat/hierarchical
#     --num_epochs 30 \
#     --train_num_workers 4 \
#     --val_num_workers 2  \
#     | tee MUTAN.txt 


############################################LIDAR########################################################
# python  train.py \
#     --model_type LIDAR_MODEL \
#     --train_batch_size 10 \
#     --model_dir ../output/LIDAR_MODEL \
#     --val_batch_size 10 \
#     --encoder_type gru \
#     --load_lidar \
#     --lr 5e-4 \
#     --grouping multi_scale \
#     --num_epochs 20 \
#     --train_num_workers 4 \
#     --val_num_workers 1 \
#     --image_features ../output/processed/vgg16_features.h5 \
#     --train_encodings ../output/processed/train_questions.h5 \
#     --val_encodings ../output/processed/val_questions.h5 \
#     --vocab ../output/processed/vocab.json \
# 	  | tee lidar_model.txt 

      #(single_scale,mulit_scale)
