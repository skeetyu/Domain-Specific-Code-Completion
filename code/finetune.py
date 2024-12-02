import argparse
import logging
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def logger_setup(log_dir, log_file):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def process_func(example, tokenizer, max_len, lang):
    input_ids, attention_mask, labels = [], [], []
    if lang == 'python':
        input = example['input']
        input = '\n'.join([line.strip() for line in input.split('\n')])
        input = input + '\n'
        input = tokenizer(input, add_special_tokens=False)
        output = example['output']
        output = '\n'.join([line.strip() for line in output.split('\n')])
        output = tokenizer(output, add_special_tokens=False)

    elif lang == 'java':
        input = example['input']
        input = tokenizer(input, add_special_tokens=False)
        # output = example['output']
        output = ' ' + example['output']
        output = tokenizer(output, add_special_tokens=False)

    input_ids = input["input_ids"] + output["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = input["attention_mask"] + output["attention_mask"] + [1]
    labels = [-100] * len(input["input_ids"]) + output["input_ids"] + [tokenizer.eos_token_id]  
    
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
        # input_ids = input_ids[-max_len:]
        # attention_mask = attention_mask[-max_len:]
        # labels = labels[-max_len:]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main(args):
    # Setup logger
    logger = logger_setup(args.log_dir, args.log_file)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    # [{"input": ..., "output": ...}, ...]
    dataset = load_dataset("json", data_files=args.dataset)['train']
    n_process_func = lambda example: process_func(example, tokenizer, args.max_len, args.lang)
    tokenized_dataset = dataset.map(n_process_func, remove_columns=['input', 'output'])
    # print(tokenized_dataset[0])
    logger.info(f'Load dataset {args.dataset} with {len(dataset)} samples')
    logger.info(f"Keys of tokenized dataset: {list(tokenized_dataset.features)}")

    eval_dataset = load_dataset("json", data_files=args.validation_dataset)['train']
    eval_tokenized_dataset = eval_dataset.map(n_process_func, remove_columns=['input', 'output'])

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map='cuda')
    # model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='cuda')
    logger.info(f'Load model {args.model_id}')

    # LoraConfig
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        inference_mode = False,
        target_modules=["q_proj", "v_proj"] # add for starcoder2
    )

    model = get_peft_model(model, lora_config)
    logger.info(model.print_trainable_parameters())

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True
    )

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set Training Args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,     # ori:1e-4
        num_train_epochs=20,    # ori:20
        logging_dir=output_dir,
        logging_strategy="steps",
        logging_steps=50,       # or 50
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_tokenized_dataset
    )

    logger.info(training_args)
    logger.info("Start training ...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="The input data file path.")
    parser.add_argument("--validation_dataset", default=None, type=str, required=True,
                        help="The input validation data file path.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The directory where the finetuned model ckpt will be stored")
    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="The directory where logs will be written")
    parser.add_argument("--log_file", default=None, type=str, required=True,
                        help="The file where logs will be written")
    parser.add_argument("--model_id", type=str,
                        help="The directory where the original model is stored")
    parser.add_argument("--tokenizer_id", type=str,
                        help="The directory where the tokenizer config is stored")
    
    parser.add_argument("--max_len", type=int,
                        help="The max length of input in dataset")
    
    parser.add_argument("--lang", default='python', type=str,
                        help="The language for the code dataset")
    
    args = parser.parse_args()
    main(args)