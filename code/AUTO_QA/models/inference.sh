export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1


python  inference.py \
    --model_type MUTAN \
    --model_dir /data/ksumit/testing2/output/MUTAN_hierarchical \
    --model_name=MUTAN_lstm_Ep28.pkl \
    --test_batch_size 100 \
    --encoder_type lstm \
    --test_num_workers 1 \
    --image_features /data/ksumit/testing/output/processed/vgg16_features.h5 \
    --fusion_type hierarchical \
    --test_encodings /data/ksumit/testing2/output/processed/val_questions2.h5 \
    --vocab /data/ksumit/testing2/output/processed/vocab2.json \
    | tee test_mutan.txt

