export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2


#only for hierarchical model

python  visualize.py \
    --model_type SAN \
    --model_dir ../output/SAN \
    --model_name=SAN_gru_Ep29.pkl \
    --save_dir=../../output/results_images_SAN \
    --val_batch_size 100 \
    --encoder_type gru \
    --val_num_workers 1 \
    --image_features ../output/processed/vgg16_train_features.h5 \
    --val_encodings ../output/processed/val_questions.h5 \
    --vocab ../../output/processed/vocab_train.json \
    | tee log_san.txt


# python  visualize.py \
#     --model_type DAN \
#     --model_dir ../../output/DAN_vgg16_hierarchical \
#     --model_name=DAN_gru_Ep27.pkl \
#     --save_dir=../../output/images_DAN_vgg16 \
#     --val_batch_size 100 \
#     --encoder_type gru \
#     --val_num_workers 1 \
#     --image_features ../../output/processed/vgg16_train_features.h5 \
#     | tee log_dan.txt



# python  visualize.py \
#     --model_type MCB \
#     --model_dir ../../output/MCB_vgg16_hierarchical \
#     --model_name=MCB_gru_Ep19.pkl \
#     --save_dir=../../output/images_MCB_vgg16_hierarchical \
#     --val_batch_size 100 \
#     --encoder_type gru \
#     --val_num_workers 1 \
#     --image_features ../../output/processed/vgg16_train_features.h5 \
#     | tee log_mcb_hierarchical_1.txt


# python  visualize.py \
#     --model_type MFB \
#     --model_dir ../../output/MFB_vgg16_hierarchical \
#     --model_name=MFB_gru_Ep19.pkl \
#     --save_dir=../../output/images_MFB_vgg16_hierarchical \
#     --val_batch_size 100 \
#     --encoder_type gru \
#     --val_num_workers 1 \
#     --image_features ../../output/processed/vgg16_train_features.h5 \
#     | tee log_mfb_hierarchical.txt




# python  visualize.py \
#     --model_type MLB \
#     --model_dir ../../output/MLB_hierarchical \
#     --model_name=MLB_lstm_Ep9.pkl \
#     --save_dir=../../output/images_MLB_vgg16_hierarchical \
#     --val_batch_size 100 \
#     --encoder_type lstm \
#     --val_num_workers 1 \
#     --image_features ../../output/processed/vgg16_train_features.h5 \
#     | tee log_mlb_hierarchical.txt

# python  visualize.py \
#     --model_type MUTAN \
#     --model_dir ../../output/MUTAN_hierarchical \
#     --model_name=MUTAN_lstm_Ep28.pkl \
#     --save_dir=../../output/images_MUTAN_vgg16_hierarchical \
#     --val_batch_size 100 \
#     --encoder_type lstm \
#     --val_num_workers 1 \
#     --image_features ../../output/processed/vgg16_train_features.h5 \
#     | tee log_mutan_hierarchical.txt



