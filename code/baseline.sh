DOMAIN=Spring
CUDA=0
LANG=java       # java or python

python -u run.py \
        --dataset=../dataset/completion/${DOMAIN}.json \
        --output_dir=../prediction/${DOMAIN} \
        --predictions=6.7b \
        --log_dir=log/baseline \
        --log_file=${DOMAIN}-6.7b.log \
        --large_model_ckpt=../deepseek/deepseek-coder-6.7b-base \
        --large_model_type=deepseek-coder \
        --cuda=${CUDA} \
        --lang=${LANG} \
        --tokenizer_dir=../deepseek/deepseek-coder-6.7b-base \
        --baseline \
        --max_gen_len=128
