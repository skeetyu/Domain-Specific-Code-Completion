DOMAIN=Spring
FT_CKPT=1125
CLASSIFIER_CKPT=10
CUDA=0
LANG=java

python -u run.py \
        --dataset=../dataset/completion/${DOMAIN}.json \
        --output_dir=../prediction/${DOMAIN} \
        --predictions=csd \
        --log_dir=log/csd \
        --log_file=${DOMAIN}.log \
        --corpus=../dataset/corpus/${DOMAIN}.txt \
        --small_model_ckpt=../deepseek/deepseek-coder-1.3b-base-GPTQ \
        --small_model_type=deepseek-coder \
        --small_model_lora=../deepseek/ft-${DOMAIN}/checkpoint-${FT_CKPT} \
        --large_model_ckpt=../deepseek/deepseek-coder-6.7b-base \
        --large_model_type=deepseek-coder \
        --tokenizer_dir=../deepseek/deepseek-coder-6.7b-base \
        --cuda=${CUDA} \
        --lang=${LANG} \
        --max_gen_len=128 \
        --max_draft_len=8 \
        --with_sd \
        --with_classifier \
        --classifier_ckpt=../classifier/models/${DOMAIN}/ckpt${CLASSIFIER_CKPT}.pth \
        --classifier_input_size=57 \
        --classifier_scale_size=16 \
        --th_classifier_prob=0.5