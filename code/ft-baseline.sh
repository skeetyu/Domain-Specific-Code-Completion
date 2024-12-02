DOMAIN=Spring
FT_CKPT=1125
CUDA=0
LANG=java       # java or python

python -u run.py \
        --dataset=../dataset/completion/${DOMAIN}.json \
        --output_dir=prediction/${DOMAIN} \
        --predictions=1.3b-ft-${FT_CKPT} \
        --log_dir=log/baseline \
        --log_file=${DOMAIN}-1.3b-ft.log \
        --large_model_ckpt=../deepseek/deepseek-coder-1.3b-base-GPTQ \
        --large_model_lora=../deepseek/ft-${DOMAIN}/checkpoint-${FT_CKPT} \
        --large_model_type=deepseek-coder \
        --large_model_bf16 \
        --cuda=${CUDA} \
        --lang=${LANG} \
        --tokenizer_dir=../deepseek/deepseek-coder-6.7b-base \
        --baseline \
        --max_gen_len=128
