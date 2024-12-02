export CUDA_VISIBLE_DEVICES=0
DOMAIN=Spring
LANG=java     # java or python

python -u finetune.py \
        --dataset=../dataset/finetune/${DOMAIN}-ft.json \
        --validation_dataset=../dataset/finetune/${DOMAIN}-ft-val.json \
        --output_dir=../deepseek/ft-${DOMAIN} \
        --log_dir=log/finetune \
        --log_file=ft-${DOMAIN}.log \
        --model_id=../deepseek/deepseek-coder-1.3b-base-GPTQ \
        --tokenizer_id=../deepseek/deepseek-coder-6.7b-base \
        --max_len=1024 \
        --lang=${LANG}