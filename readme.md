# README

In this project, we propose a collaborative inference framework incorporating the speculative decoding algorithm to effectively combine the completion results of large and small code models with a well-designed classifier, for better domain-specific code completion tasks.


## Project structure

- **code**: The implementation code and script files.
- **dataset**: The directory for storing the dataset.
- **deepseek**: The directory for storing the code models.
- **env.yaml**: The environment configuration.


## Train and evaluate

The workflow of the entire framework mainly consists of the following four steps:

### 1. Fine-tune the small code model

run `finetune.sh`.

Please ensure the following args with your actual paths:
    - **dataset**, **validation_dataset**: to load data to fine-tune the model
    - **output_dir**: to save your fine-tuned model
    - **model_id**, **tokenizer_id**: to load your small model

### 2. Collect training data for the classifier

run `collect.sh` to collect the training and validation data respectively.

Please ensure the following args with your actual paths:
    - **dataset**: to load data
    - **output_dir**: to save training data for the classfier
    - **corpus**: to load the code corpus
    - **small_model_ckpt**, **small_model_lora**, **large_model_ckpt**, **tokenizer_id**: to load the large and small model

### 3. Train the classifier

run `train.sh`.

Please ensure the following args with your actual paths:
- **train_dir**, **validation_dir**: to load the training data
- **output_dir**: to save the classifier model

### 4. Inference

#### Large model

run `baseline.sh`

Please ensure the following args with your actual paths:
    - **dataset**: to load the data for completion
    - **output_dir**: to save the prediction results
    - **large_model_ckpt**, **tokenizer_dir**: to load the large model

#### Small model

run `ft-baseline.sh`

Please ensure the following args with your actual paths:
    - **dataset**: to load the data for completion
    - **output_dir**: to save the prediction resulsts
    - **large_model_ckpt**, **large_model_lora**, **tokenizer_dir**: to load the fine-tuned small model

#### Our approach

run `csd.sh`
Please ensure the following args with your actual paths:
    - **dataset**: to load the data for completion
    - **output_dir**: to save the prediction resulsts
    - **corpus**: to load the code corpus
    - **small_model_ckpt**, **small_model_lora**, **large_model_ckpt**, **large_model_lora**, **tokenizer_dir**: to load the fine-tuned small model and large model
    - **classifier_ckpt**: to load the classifier model