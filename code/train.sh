DOMAIN=Spring
CUDA=0

python -u train.py \
        --train_dir=../classifier/${DOMAIN}-train \
        --validation_dir=../classifier/${DOMAIN}-val \
        --output_dir=../classifier/models/${DOMAIN} \
        --log_dir=log/classifier \
        --log_file=${DOMAIN}.log \
        --cuda=${CUDA} \
        --seed=42 \
        --model_type=deepseek-coder \
        --epoch=30 \
        --save_step=1 \
        --batch_size=64 \
        --lr=0.0005 \
        --dropout=0.5 \
        --weight_decay=0.00001 \
        --lr_step=10 \
        --lr_gamma=0.5 \
        --input_size=57 \
        --scale_size=16 \
        --do_train \
        --threshold=0.5