DOMAIN=Spring
TYPE=train      # train or val
FT_CKPT=1125
CUDA=0
LANG=java       # java or python

python -u run.py \
        --dataset=../dataset/classifier/${DOMAIN}-classifier-${TYPE}.json \
        --output_dir=../classifier/${DOMAIN}-${TYPE} \
        --log_dir=log/collect \
        --log_file=${DOMAIN}-${TYPE}.log \
        --corpus=../dataset/corpus/${DOMAIN}.txt \
        --small_model_ckpt=../deepseek/deepseek-coder-1.3b-base-GPTQ \
        --small_model_type=deepseek-coder \
        --small_model_lora=../deepseek/ft-${DOMAIN}/checkpoint-${FT_CKPT} \
        --large_model_ckpt=../deepseek/deepseek-coder-6.7b-base \
        --large_model_type=deepseek-coder \
        --tokenizer_dir=../deepseek/deepseek-coder-6.7b-base \
        --cuda=${CUDA} \
        --lang=${LANG} \
        --collect_classifier_data 
