# CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 4567 --wait-for-client prepare.py \
CUDA_VISIBLE_DEVICES=1 python prepare.py \
    --config_name gated_encoder_model \
    --inputter_name strat \
    --train_input_file my_train.txt \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --use_all_persona False \
    --encode_context True \
    --single_processing 
