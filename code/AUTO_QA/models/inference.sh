export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1


python  inference.py \
    --model_type MUTAN \
    --model_dir ../output/MUTAN_hierarchical \
    --model_name=MUTAN_lstm_Ep28.pkl \
    --test_batch_size 100 \
    --encoder_type lstm \
    --test_num_workers 1 \
    --image_features ../output/processed/vgg16_test_features.h5 \
    --fusion_type hierarchical \
    --test_encodings ../output/processed/test_questions.h5 \
    --vocab ../output/processed/vocab_test.json \
    | tee test_mutan.txt

