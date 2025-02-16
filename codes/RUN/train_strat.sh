CUDA_VISIBLE_DEVICES=0 python train.py \
    --config_name gated_encoder_model \
    --inputter_name strat \
    --eval_input_file  my_valid.txt \
    --seed 13 \
    --max_input_length 512 \
    --max_decoder_input_length 50 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 1.5e-5 \
    --num_epochs 9 \
    --warmup_steps 0 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true \
    --use_all_persona False \
    --encode_context True

# CUDA_VISIBLE_DEVICES=1 python train.py \
# my_valid.txt  gated_dpr_model /data/wanghongkai/ds/PAL-main/codes/valid.txt
# python -m debugpy --listen 4567 --wait-for-client train.py \