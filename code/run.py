import os
import time
import argparse
import logging
import json
import torch

from tqdm import tqdm

from decoding import Generator, Bild_Generator, Raw_SD_Generator
from classifier import ClassifierDataCollector, Classifier
from utils import set_seed

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fuzzywuzzy import fuzz
from auto_gptq import exllama_set_max_input_length

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

def main(args):
    # Setup logger
    logger = logger_setup(args.log_dir, args.log_file)
    logger.info("Training/Evaluation parameters %s", args)

    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    args.cuda = "0"

    # Load dataset
    with open(args.dataset, 'r') as f:
        eval_dataset = f.readlines()
    if args.corpus is not None:
        with open(args.corpus, 'r') as f:
            corpus = f.readlines()
        logger.info(f"Loading corpus from {args.corpus}")

    device = torch.device(f'cuda:{args.cuda}')

    # Load large model
    logger.info(f'Loading large model: {args.large_model_ckpt}')
    if args.large_model_type in ['deepseek-coder', 'starcoder2']:
        if args.large_model_bf16:
            large_model = AutoModelForCausalLM.from_pretrained(args.large_model_ckpt, torch_dtype=torch.bfloat16, device_map=device)
        else:
            large_model = AutoModelForCausalLM.from_pretrained(args.large_model_ckpt, device_map=device)

        if args.large_model_lora:
            logger.info(f'Loading large model\'s adapter from {args.large_model_lora}')
            large_model = PeftModel.from_pretrained(large_model, args.large_model_lora, torch_device=device)

        large_model = large_model.to(device)
        large_model.eval()
    else:
        raise NotImplementedError('Load not implemented model')

    # Load small model
    if args.collect_classifier_data or args.with_sd or args.bild or args.raw_sd:
        logger.info(f'Loading small model: {args.small_model_ckpt}')
        if args.small_model_type in ['deepseek-coder', 'starcoder2']:
            # small_model = AutoModelForCausalLM.from_pretrained(args.small_model_ckpt, torch_dtype=torch.bfloat16, device_map=device)

            small_model = AutoModelForCausalLM.from_pretrained(args.small_model_ckpt, torch_dtype=torch.float16, device_map=device)    # for auto-gptq

            small_model = exllama_set_max_input_length(small_model, max_input_length=8192)

            if args.small_model_lora:
                logger.info(f'Loading small model\'s adapter from {args.small_model_lora}')
                small_model = PeftModel.from_pretrained(small_model, args.small_model_lora, torch_device=device)

            small_model = small_model.to(device)
            small_model.eval()
        else:
            raise NotImplementedError('Load not implemented model')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Collect classifier training data
    if args.collect_classifier_data:
        logger.info(f'Collecting classifier\'s training datas...')
        logger.info(f'Evaluating {len(eval_dataset)} cases...')

        collector = ClassifierDataCollector(
            l_model=large_model,
            l_model_type=args.large_model_type,
            s_model=small_model,
            s_model_type=args.small_model_type,
            tokenizer=tokenizer,
            device=device
        )

        collector.collect_parallel(
            dataset=eval_dataset,
            output_dir=args.output_dir,
            corpus=corpus,
            lang=args.lang
        )

        logger.info(f'Write embeddings_xx.pt and features.json in {args.output_dir}')

    # Evaluation with large model
    if args.baseline:
        generator = Generator(
            l_model=large_model,
            l_model_type=args.large_model_type,
            tokenizer=tokenizer,
            device=device,
            lang=args.lang,
            corpus=None,
            with_sd=False,
            s_model=None,
            s_model_type=None,
            with_classifier=False,
            classifier=None,
            collector=None
        )

        preds = []
        logger.info(f'Evaluating {len(eval_dataset)} cases with baseline_decoding ...')

        es = 0
        em = 0
        total = len(eval_dataset)

        for eval_case in tqdm(eval_dataset):
            eval_case = json.loads(eval_case)
            input = eval_case['input'].replace(" <EOL> ", "\n")
            input_ids = tokenizer(input, return_tensors="pt").input_ids

            gt = eval_case['gt'] + '\n'

            pred = generator.baseline_decoding(
                input_ids=input_ids,
                max_gen_len=args.max_gen_len
            )
            preds.append(pred)
            # print(pred)

            es += fuzz.ratio(gt.strip(), pred.strip())
            em += 1 if gt.strip() == pred.strip() else 0

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = os.path.join(args.output_dir, f'{args.predictions}_baseline.txt')
        with open(output_file, 'w') as wf:
            for pred in preds:
                wf.write(f'{pred}\n')
        logger.info(f'Write predictions in {output_file}')
        logger.info(f'EM_cnt: {em}, Edit Sim: {es/total}, EM: {em/total}')

    # # sd
    # if args.with_sd and not args.with_classifier:
    #     generator = Generator(
    #         l_model=large_model,
    #         l_model_type=args.large_model_type,
    #         tokenizer=tokenizer,
    #         device=device,
    #         lang=args.lang,
    #         corpus=corpus,
    #         with_sd=True,
    #         s_model=small_model,
    #         s_model_type=args.small_model_type,
    #         with_classifier=False,
    #         classifier=None,
    #         collector=None
    #     )

    #     preds = []
    #     logger.info(f'Evaluating {len(eval_dataset)} case with speculative_decoding ...')

    #     es = 0
    #     em = 0
    #     total = len(eval_dataset)

    #     for eval_case in tqdm(eval_dataset):
    #         eval_case = json.loads(eval_case)
    #         input = eval_case['input'].replace(" <EOL> ", "\n")
    #         input_ids = tokenizer(input, return_tensors="pt").input_ids

    #         gt = eval_case['gt'] + '\n'
    #         gt_ids = tokenizer(gt, return_tensors="pt").input_ids[:, 1:].to(device)   # get rid of bos_token

    #         t = time.perf_counter()
    #         pred = generator.speculative_decoding(
    #             input_ids=input_ids,
    #             max_gen_len=args.max_gen_len,
    #             max_draft_len=args.max_draft_len,
    #             th_classifier_prob=None
    #         )
    #         print(f'speculative_decoding cost: {time.perf_counter() - t:.8f}s')
    #         preds.append(pred)

    #         es += fuzz.ratio(gt.strip(), pred.strip())
    #         em += 1 if gt.strip() == pred.strip() else 0

    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir)
    #     output_file = os.path.join(args.output_dir, f'{args.predictions}_sd.txt')
    #     with open(output_file, 'w') as wf:
    #         for pred in preds:
    #             wf.write(f'{pred}\n')
    #     logger.info(f'Write predictions in {output_file}')
    #     logger.info(f'EM_cnt: {em}, Edit Sim: {es/total}, EM: {em/total}')
    
    # csd
    if args.with_sd and args.with_classifier:
        collector = ClassifierDataCollector(
            l_model=large_model,
            l_model_type=args.large_model_type,
            s_model=small_model,
            s_model_type=args.small_model_type,
            tokenizer=tokenizer,
            device=device
        )

        classifier = Classifier(
            input_size=args.classifier_input_size,
            scale_size=args.classifier_scale_size,
            dropout=0.2,
            train_dir=None,
            validation_dir=None,
            model_path=args.classifier_ckpt,
            model_type=args.large_model_type,
            device=device,
            logger=logger
        )

        generator = Generator(
            l_model=large_model,
            l_model_type=args.large_model_type,
            tokenizer=tokenizer,
            device=device,
            lang=args.lang,
            corpus=corpus,
            with_sd=True,
            s_model=small_model,
            s_model_type=args.small_model_type,
            with_classifier=True,
            classifier=classifier,
            collector=collector
        )

        preds = []
        logger.info(f'Evaluating {len(eval_dataset)} case with speculative_decoding with classifier ...')

        es = 0
        em = 0
        total = len(eval_dataset)

        for eval_case in tqdm(eval_dataset):
            eval_case = json.loads(eval_case)
            input = eval_case['input'].replace(" <EOL> ", "\n")
            input_ids = tokenizer(input, return_tensors="pt").input_ids

            gt = eval_case['gt']
            gt = ' ' + gt       # 增加额外的空格或\n都是为了统计，不影响实际预测
            if args.lang == 'python':
                gt = gt + '\n'    
            
            if args.large_model_type == 'deepseek-coder':
                gt_ids = tokenizer(gt, return_tensors="pt").input_ids[:, 1:].to(device)   # get rid of bos_token
            # elif args.large_model_type == 'starcoder2':
            #     gt_ids = tokenizer(gt, return_tensors="pt").input_ids.to(device)

            pred = generator.speculative_decoding(
                input_ids=input_ids,
                max_gen_len=args.max_gen_len,
                max_draft_len=args.max_draft_len,
                th_classifier_prob=args.th_classifier_prob,
            )
            preds.append(pred)

            es += fuzz.ratio(gt.strip(), pred.strip())
            em += 1 if gt.strip() == pred.strip() else 0

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        output_file = os.path.join(args.output_dir, f'{args.predictions}_csd_{args.th_classifier_prob}.txt')
        with open(output_file, 'w') as wf:
            for pred in preds:
                wf.write(f'{pred}\n')
        logger.info(f'Write predictions in {output_file}')
        logger.info(f'EM_cnt: {em}, Edit Sim: {es/total}, EM: {em/total}')

    if args.bild:
        generator = Bild_Generator(
            l_model=large_model,
            l_model_type=args.large_model_type,
            tokenizer=tokenizer,
            device=device,
            lang=args.lang,
            s_model=small_model,
            s_model_type=args.small_model_type,
            fallback_th=args.fallback_th,
            rollback_th=args.rollback_th
        )

        preds = []
        logger.info(f'Evaluating {len(eval_dataset)} case with bild methods ...')

        es = 0
        em = 0
        total = len(eval_dataset)

        for eval_case in tqdm(eval_dataset):
            eval_case = json.loads(eval_case)
            input = eval_case['input'].replace(" <EOL> ", "\n")
            input_ids = tokenizer(input, return_tensors="pt").input_ids
            
            gt = eval_case['gt']

            pred = generator.sd_bild(
                input_ids=input_ids,
                max_gen_len=args.max_gen_len,
                max_draft_len=args.max_draft_len
            )
            preds.append(pred)

            es += fuzz.ratio(gt.strip(), pred.strip())
            em += 1 if gt.strip() == pred.strip() else 0

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        if args.fallback_th is not None and args.rollback_th is not None:
            output_file = os.path.join(args.output_dir, f'{args.predictions}_{args.fallback_th}_{args.rollback_th}.txt')
        else:
            output_file = os.path.join(args.output_dir, f'{args.predictions}.txt')

        with open(output_file, 'w') as wf:
            for pred in preds:
                wf.write(f'{pred}\n')
        logger.info(f'Write predictions in {output_file}')
        logger.info(f'EM_cnt: {em}, Edit Sim: {es/total}, EM: {em/total}')

    if args.raw_sd:
        generator = Raw_SD_Generator(
            l_model=large_model,
            l_model_type=args.large_model_type,
            tokenizer=tokenizer,
            device=device,
            lang=args.lang,
            s_model=small_model,
            s_model_type=args.small_model_type,
        )

        preds = []
        logger.info(f'Evaluating {len(eval_dataset)} case with raw sd methods ...')

        es = 0
        em = 0
        total = len(eval_dataset)

        for eval_case in tqdm(eval_dataset):
            eval_case = json.loads(eval_case)
            input = eval_case['input'].replace(" <EOL> ", "\n")
            input_ids = tokenizer(input, return_tensors="pt").input_ids
            
            gt = eval_case['gt']

            pred = generator.speculative_decoding(
                input_ids=input_ids,
                max_gen_len=args.max_gen_len,
                max_draft_len=args.max_draft_len
            )
            preds.append(pred)

            es += fuzz.ratio(gt.strip(), pred.strip())
            em += 1 if gt.strip() == pred.strip() else 0

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        output_file = os.path.join(args.output_dir, f'{args.predictions}.txt')

        with open(output_file, 'w') as wf:
            for pred in preds:
                wf.write(f'{pred}\n')
        logger.info(f'Write predictions in {output_file}')
        logger.info(f'EM_cnt: {em}, Edit Sim: {es/total}, EM: {em/total}')

    logger.info('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, type=str, required=True,
                        help="The input data file path")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions or collected dataset will be stored")
    parser.add_argument("--predictions", default=None, type=str,
                        help="Prefix of the output file where the predictions will be stored")
    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="The directory where logs will be written")
    parser.add_argument("--log_file", default=None, type=str, required=True,
                        help="The path where logs will be written")
    parser.add_argument("--corpus", type=str,
                        help="The file where the training code corpus is stored")
    
    parser.add_argument("--seed", default=42, type=int)
    
    parser.add_argument("--small_model_ckpt", type=str,
                        help="The directory where stores the small model")
    parser.add_argument("--small_model_type", type=str,
                        choices=['deepseek-coder', 'starcoder2', 'codellama'], 
                        help="The type of small model")
    parser.add_argument("--small_model_lora", type=str, default=None,
                        help="The directory where the peft adapter for small model is stored")
    
    parser.add_argument("--large_model_ckpt", type=str, required=True,
                        help="The directory where stores the large model")
    parser.add_argument("--large_model_type", type=str, required=True,
                        choices=['deepseek-coder', 'starcoder2', 'codellama'],
                        help="The type of large model")
    parser.add_argument("--large_model_lora", type=str, default=None,
                        help="The directory where the peft adapter for large model is stored")
    parser.add_argument("--large_model_bf16", action="store_true",
                        help="Whether to use bfloat16 dtype to load the large model")
    
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="The directory where stores the tokenizer")
    parser.add_argument("--cuda", type=int, default=2,
                        help="The device where loads the models")
    
    parser.add_argument("--lang", type=str, required=True, choices=['python', 'java'],
                        help="The language of the code to be completed")
    parser.add_argument("--max_gen_len", default=64, type=int,
                        help="Max length of generated codes")
    parser.add_argument("--max_draft_len", default=8, type=int,
                        help="Max length of drafted tokens generated by small model")
    
    parser.add_argument("--baseline", action="store_true",
                        help="Use standard auto-regressive generation")
    parser.add_argument("--collect_classifier_data", action="store_true",
                        help="Whether to collect the data for classifier")
    
    parser.add_argument("--with_sd", action="store_true",
                        help="Use speculative decoding generation")
    parser.add_argument("--with_classifier", action="store_true",
                        help="Whether to use classifier during speculative decoding")
    parser.add_argument("--th_classifier_prob", default=0.5, type=float,
                        help="Threshold to determine whether a token is domain specific")
    parser.add_argument("--classifier_ckpt", type=str,
                        help="The location of classifier model pth")
    parser.add_argument("--classifier_input_size", type=int,
                        help="The input size of the classifier MLP network")
    parser.add_argument("--classifier_scale_size", type=int,
                        help="The scale size of the classifier MLP network")

    parser.add_argument("--bild", action="store_true",
                        help="Whether to run baseline of bild method")
    parser.add_argument("--fallback_th", type=float,
                        help="Fallback threshold for bild method")
    parser.add_argument("--rollback_th", type=float,
                        help="Rollback threshold for bild method")
    
    parser.add_argument("--raw_sd", action="store_true",
                        help="Whether to run baseline of raw sd methods with acceptance with certain probabilities")

    args = parser.parse_args()

    if (args.baseline or args.with_sd) and args.predictions is None:
        raise ValueError("Must set predictions when evulating baseline or speculative decoding")

    if args.with_sd and (args.small_model_ckpt is None or args.small_model_type is None):
        raise ValueError("The value of small model ckpt and type can not be none when setting --with_sd")
        
    if args.with_classifier and not args.with_sd:
        raise ValueError("Must set --with_sd when setting --with_classifier")

    if args.with_classifier and args.classifier_ckpt is None:
        raise ValueError("The value of classifier ckpt can not be none when setting --with_classifier")

    main(args)