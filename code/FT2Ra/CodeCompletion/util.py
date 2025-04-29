import json
import os
import random
import re

from fuzzywuzzy import fuzz
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer)

from dataset import LineCompletionDataset, TokenCompletionDataset, TokenCompletionDatasetSpecial, \
    TokenCompletionDatasetReacc, TokenCompletionDatasetReaccTrain
from model import CodeCompletionModel, CodeCompletionModel_Gpt2, CodeCompletionModel_Llama, CodeCompletionModel_StarCoder


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'llama': (AutoConfig, AutoModelForCausalLM, AutoTokenizer),  # add myself
    'starcoder': (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
}

def get_knm_base_ratio(modeltype,is_finetuned,data_name):
    # 初始化数据
    data = [
        [0.4698546, 0.552237061, 0.501065913, 0.527554919, 0.487092032, 0.534013211, 0.647030006, 0.600645203,
         0.566932073, 0.595349672, 0.649513591, 0.524072363],
        [0.750626723, 0.70849528, 0.730193605, 0.701337154, 0.73660688, 0.675049253, 0.748805879, 0.761383073,
         0.738989798, 0.729881836, 0.773011853, 0.769290812],
        [0.425858611, 0.546527818, 0.471481401, 0.509972866, 0.450102911, 0.495654189, 0.629067157, 0.569494604,
         0.541054989, 0.587877979, 0.649513591, 0.524072363],
        [0.756267235, 0.705488194, 0.721318251, 0.700575998, 0.746251103, 0.674933364, 0.771014493, 0.76964166,
         0.737248072, 0.727922035, 0.773011853, 0.769290812]
    ]

    head = ["rest-assured", "AmazeFileManager", "dropwizard", "eureka", "feign", "galaxy", "interview",
            "logging-log4j1", "requery", "Froyo_Email", "javaCorpus", "py150"]
    modeltypes = ["gpt2", "gpt2"]
    is_finetuneds = [False, True]

    # 创建字典
    result = {}

    for i in range(4):
        for j in range(12):
            key = (modeltypes[i], is_finetuneds[i], head[j])
            result[key] = data[i][j]
    # 输出字典
    # print(result)
    return result[(modeltype,is_finetuned,data_name)]


def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens

def get_dataloader_line(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir):
    cache_dir = os.path.join(save_dir, "dataCache")
    #  https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-line/code/dataset.py#L242
    # 这里不是1024，而是924，因为gpt2的输入长度是1024，但是这里要预留100个位置给ground truth

    # block_size = 924
    if modeltype == 'gpt2':
        block_size = 924
    elif modeltype == 'llama':
        block_size = 4096   # update myself
    elif modeltype == 'starcoder':
        block_size = 4608   # update myself

    data_tag = modeltype + "_" + dataname+ "_" + datatype + "_line"
    assert datatype in ["test","testreacc","testbm25"], "datatype should be test"
    if dataname == "javaCorpus":
        if datatype == "test":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/java/test.json"
        elif datatype == "testreacc":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/java/test_reacc.json"
        elif datatype == "testbm25":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/java/test_bm25.json"
        else:
            raise NotImplementedError
    elif dataname == "py150":
        if datatype == "test":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test.json"
        elif datatype == "testreacc":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_reacc.json"
        elif datatype == "testbm25":
            datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_bm25.json"
        else:
            raise NotImplementedError
    elif dataname == "ten2one":
        datafile = os.path.join(datadir, "large_projects", datatype + "_line.txt")
    else:
        if datatype == "testbm25":
            datafile = os.path.join(datadir,"large_projects", dataname, "test_bm25_line.txt")
        elif datatype == "testreacc":
            datafile = os.path.join(datadir,"large_projects", dataname, "test_reacc_line.txt")
        else:
            datafile = os.path.join(datadir,"large_projects", dataname, datatype + "_line.txt")
    my_dataset = LineCompletionDataset(tokenizer, cache_dir, datafile , block_size, data_tag)
    my_sampler = SequentialSampler(my_dataset)  # if args.local_rank == -1 else DistributedSampler(eval_dataset)
    my_dataloader = DataLoader(my_dataset, sampler=my_sampler, batch_size=batch_size, drop_last=True)
    return my_dataloader,my_dataset
# def get_dataloader_special_line(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir):
#     cache_dir = os.path.join(save_dir, "dataCache")
#     block_size = 1024
#     data_tag = modeltype + "_" + dataname+ "_" + datatype + "_line"
#     assert datatype in ["test"], "datatype should be test"
#     datafile = os.path.join(datadir,"large_projects", dataname, datatype + "_line.txt")
#     my_dataset = LineCompletionDataset(tokenizer, cache_dir, datafile , block_size, data_tag)
#     my_sampler = SequentialSampler(my_dataset)  # if args.local_rank == -1 else DistributedSampler(eval_dataset)
#     my_dataloader = DataLoader(my_dataset, sampler=my_sampler, batch_size=batch_size, drop_last=True)
#     return my_dataloader,my_dataset
def get_dataloader_token_reacc_train(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir):
    cache_dir = os.path.join(save_dir, "dataCache")
    block_size = 1024
    data_tag = modeltype + "_" + dataname + "_" + datatype
    assert datatype in ["trainreacc"], "datatype should be testreacc"
    # /data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_reacc.json
    datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_reacc.json"
    my_dataset = TokenCompletionDatasetReaccTrain(tokenizer, cache_dir, datafile, block_size, data_tag)
    my_sampler = SequentialSampler(my_dataset)
    my_dataloader = DataLoader(my_dataset, sampler=my_sampler, batch_size=batch_size, drop_last=True)
    return my_dataloader, my_dataset
def get_dataloader_token_reacc(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir):
    cache_dir = os.path.join(save_dir, "dataCache")
    block_size = 1024
    data_tag = modeltype + "_" + dataname+ "_" + datatype
    assert datatype in ["testreacc"], "datatype should be testreacc"
    # /data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_reacc.json
    datafile = "/data/c/base/RAPT/CodeCompletion/dataset/lineTestdata/python/test_reacc.json"
    my_dataset = TokenCompletionDatasetReacc(tokenizer, cache_dir, datafile , block_size, data_tag)
    my_sampler = SequentialSampler(my_dataset)
    my_dataloader = DataLoader(my_dataset, sampler=my_sampler, batch_size=batch_size, drop_last=True)
    return my_dataloader,my_dataset

def get_dataloader_token(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir):
    cache_dir = os.path.join(save_dir, "dataCache")
    block_size = 1024
    data_tag = modeltype + "_" + dataname+ "_" + datatype
    assert datatype in ["train", "test", "dev"], "datatype should be train or test or dev"
    # /data/c/base/RAPT/CodeCompletion/dataset/javaCorpus/token_completion
    if dataname == "javaCorpus":
        datafile = os.path.join(datadir,"javaCorpus/token_completion", datatype + ".txt")
    elif dataname == "py150":
        datafile = os.path.join(datadir,"py150/token_completion", datatype + ".txt")
    elif dataname == "ten2one":
        datafile = os.path.join(datadir, "large_projects", datatype + ".txt")
    else:
        datafile = os.path.join(datadir, "large_projects", dataname, datatype + ".txt")
    if dataname in ["javaCorpus","py150"]:
        my_dataset = TokenCompletionDataset(tokenizer, cache_dir, datafile , block_size, data_tag)
    else:
        my_dataset = TokenCompletionDatasetSpecial(tokenizer, cache_dir, datafile, block_size, data_tag)
    my_sampler = SequentialSampler(my_dataset)
    my_dataloader = DataLoader(my_dataset, sampler=my_sampler, batch_size=batch_size, drop_last=True)
    return my_dataloader,my_dataset

def get_online_model_knm(modeltype,pretrained,knn_save_dir):
    # 加载knn索引，方便online时候进行knn
    lit_file = "./py_literals.json"    # myself
    do_lower_case = False
    do_lower_case = True
    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[modeltype]
    # get special tokens
    special_tokens = get_special_tokens(lit_file)

    config = config_class.from_pretrained(pretrained)

    # update myself
    if modeltype == 'llama':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>')
    elif modeltype == 'starcoder':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>', pad_token='<pad>')
    elif modeltype == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained(pretrained,
                                                    do_lower_case=do_lower_case, sep_token='<EOL>',
                                                    bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                    unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens,
                                                    timeout=200)

    # 只评估，不训练，把模型初始化参数设置为0，这样保证每次预测的结果都是一样的
    config.initializer_range = 0
    if modeltype == "gpt2":
        vecs_path = os.path.join(knn_save_dir, "knm_vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knm_knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knm_knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")
        model = CodeCompletionModel_Gpt2(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)

    # add myself
    elif modeltype == "llama":
        vecs_path = os.path.join(knn_save_dir, "knm_vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knm_knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knm_knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")
    #     vecs_path = os.path.join(knn_save_dir, "vecs.pkl")
    #     knn_infos_path = os.path.join(knn_save_dir, "knn_infos.pkl")
    #     index_save_path = os.path.join(knn_save_dir, "knn.index")
    #     xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")

        model = CodeCompletionModel_Llama(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)

    elif modeltype == "starcoder":
        vecs_path = os.path.join(knn_save_dir, "knm_vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knm_knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knm_knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")

        model = CodeCompletionModel_StarCoder(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)


    else:
        raise NotImplementedError
    return model, tokenizer, config

def get_online_model(modeltype,pretrained,knn_save_dir):
    # 加载knn索引，方便online时候进行knn
    lit_file = "./java_literals.json"    # myself
    do_lower_case = False
    do_lower_case = True
    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[modeltype]
    # get special tokens
    special_tokens = get_special_tokens(lit_file)
    
    config = config_class.from_pretrained(pretrained)

    # update myself
    if modeltype == 'llama':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>')
    elif modeltype == 'starcoder':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>', pad_token='<pad>')

    elif modeltype == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained(pretrained,
                                                    do_lower_case=do_lower_case, sep_token='<EOL>',
                                                    bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                    unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens,
                                                    timeout=200)
    # 只评估，不训练，把模型初始化参数设置为0，这样保证每次预测的结果都是一样的

    config.initializer_range = 0
    if modeltype == "gpt2":
        vecs_path = os.path.join(knn_save_dir, "vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")
        model = CodeCompletionModel_Gpt2(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)

    # add myself
    elif modeltype == "llama":
        vecs_path = os.path.join(knn_save_dir, "vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")

        model = CodeCompletionModel_Llama(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)

    elif modeltype == 'starcoder':
        vecs_path = os.path.join(knn_save_dir, "vecs.pkl")
        knn_infos_path = os.path.join(knn_save_dir, "knn_infos.pkl")
        index_save_path = os.path.join(knn_save_dir, "knn.index")
        xgboost_path = os.path.join(knn_save_dir, "xgb_model.json")

        model = CodeCompletionModel_StarCoder(model_class, pretrained, config,tokenizer,vecs_path, knn_infos_path,index_save_path,xgboost_path)


    else:
        raise NotImplementedError
    return model, tokenizer, config

def get_model(modeltype,pretrained):
    # for fine-tuning and create knn index
    lit_file = "./py_literals.json"    # myself

    do_lower_case = False
    do_lower_case = True
    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[modeltype]
    # get special tokens
    special_tokens = get_special_tokens(lit_file)

    # update myself
    if modeltype == 'llama':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>')
    elif modeltype == 'starcoder':
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>', pad_token='<pad>')
    elif modeltype == 'gpt2':
        tokenizer = tokenizer_class.from_pretrained(pretrained,
                                                    do_lower_case=do_lower_case, sep_token='<EOL>',
                                                    bos_token='<s>', eos_token='</s>', pad_token='<pad>',
                                                    unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens,
                                                    timeout=200)
    # 只评估，不训练，把模型初始化参数设置为0，这样保证每次预测的结果都是一样的
    config = config_class.from_pretrained(pretrained)
    config.initializer_range = 0
    model = model_class.from_pretrained(pretrained, config=config)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, config

def set_seed(param):
    random.seed(param)
    np.random.seed(param)
    torch.manual_seed(param)
    torch.cuda.manual_seed_all(param)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def eval_sequence(pred_ids, inputs, tokenizer):
    # 评估预测值和真实值是否相同，pred_ids和inputs都是(batch_size,seq_len)的tensor
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    all_pred = []
    all_gt = []
    tmp_pred = []
    tmp_gt = []
    prev_pred = None
    for pred, gt in zip(pred_ids, inputs):
        pred = pred.cpu().tolist()
        gt = gt.cpu().tolist()

        tmp_gt.extend(gt)
        tmp_pred.extend(pred)
        if tokenizer.eos_token_id in gt:
            pred = tmp_pred
            gt = tmp_gt
            for i, y in enumerate(gt):
                if i == 0:
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                        all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = []
                        now_pred = []
                    else:
                        now_gt = [y]
                        now_pred = [0] if prev_pred is None else [prev_pred]
                else:
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            cur_gt = DecodeIds(now_gt).strip().split()
                            try:
                                cur_pred = DecodeIds(now_pred).strip().split()
                                if len(cur_gt) <= len(cur_pred):
                                    cur_pred = cur_pred[:len(cur_gt)]
                                else:
                                    pad_len = len(cur_gt) - len(cur_pred)
                                    cur_pred = cur_pred + ['SPACE'] * pad_len
                                all_pred.extend(cur_pred)
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.extend(cur_gt)
                            now_gt = []
                            now_pred = []
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id] \
                            or tokenizer.convert_ids_to_tokens(y).startswith("<NUM_LIT") \
                            or tokenizer.convert_ids_to_tokens(y).startswith("<STR_LIT") \
                            or tokenizer.convert_ids_to_tokens(y).startswith("<CHAR_LIT"):
                        if len(now_gt) > 0:
                            cur_gt = DecodeIds(now_gt).strip().split()
                            try:
                                cur_pred = DecodeIds(now_pred).strip().split()
                                if len(cur_gt) <= len(cur_pred):
                                    cur_pred = cur_pred[:len(cur_gt)]
                                else:
                                    pad_len = len(cur_gt) - len(cur_pred)
                                    cur_pred = cur_pred + ['SPACE'] * pad_len
                                all_pred.extend(cur_pred)
                            except IndexError:
                                all_pred.append("<SPACE>")
                            all_gt.extend(cur_gt)
                        now_gt = [y]
                        now_pred = [pred[i - 1]]
                        try:
                            all_pred.append(DecodeIds(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append("<SPACE>")
                        all_gt.append(DecodeIds(now_gt).strip())
                        now_gt = []
                        now_pred = []
                        continue
                    now_gt.append(y)
                    now_pred.append(pred[i - 1])
            tmp_pred = []
            tmp_gt = []
    assert len(all_pred) == len(all_gt)
    return all_pred, all_gt
def calc_eval_tokens(pred_ids, inputs, tokenizer):
    all_pred, all_gt = eval_sequence([pred_ids], [inputs], tokenizer)
    total = 0
    correct = 0
    for x, y in zip(all_pred, all_gt):
        if y not in ["<s>", "</s>", "<EOL>", "<pad>"]:
            total += 1
            if x == y:
                correct += 1
    return total,correct

# myself 
# 增加model_type和language参数
# 增加返回prediction
def calc_eval_lines_CodeXGlue(pred_line, gt, tokenizer,dataname, model_type, language, print_flag=False):

    # update myself
    def DecodeIds(idxs, model_type):
        if model_type == 'llama':
            codes = tokenizer.decode(idxs)
            return codes
        else:
            codes = ""
            for idx in idxs:
                to_add = tokenizer.convert_ids_to_tokens(idx)
                if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                    if not codes.endswith(" "):
                        codes += " " + to_add[1:]
                    else:
                        codes += to_add[1:]
                elif (
                        idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                                tokenizer.pad_token_id] or
                        tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
                ):
                    codes += " " + to_add + " "
                else:
                    codes += to_add
            return codes.strip(" ")

    t = pred_line.cpu().numpy()
    t = t.tolist()
    if 0 in t:
        t = t[:t.index(0)]

    # print(f"tokens = {t}")

    # update myself
    # if dataname == "py150":
    #     text = DecodeIds(t).strip("<EOL>").strip()
    # else:
    #     text = DecodeIds(t).strip("{").strip()
    text = DecodeIds(t, model_type)
    if language == 'python':
        if '\n' in text:
            text = text[:text.find('\n')]
        if '<EOL>' in text:
            text = text[:text.find('<EOL>')]
    
    if language == 'java':
        text = text.strip()
        if ';' in text:
            text = text[:text.find(';')+1]
        elif '{' in text:
            text = text[:text.find("{")+1]
        elif '}' in text:
            text = text[:text.find("}")+1]

    if '<｜end▁of▁sentence｜' in text:
        text = text[:text.find('<｜end▁of▁sentence｜')]

    # update myself
    # gt = post_process(gt.strip())
    # text = post_process(text.strip())
    gt = gt.strip()
    text = text.strip()

    # print(f'gt = {gt}')
    # print(f'text = {text}')
    
    edit_sim = fuzz.ratio(text, gt)
    EM = 1 if text == gt else 0
    # if print_flag or EM == 1:
    #     print("pred:",text)
    #     # print("gt:",gt)
    #     print("EM:",EM)
    return EM, edit_sim, text

def show_pred(pred_line, tokenizer):
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    t = pred_line.cpu().numpy()
    t = t.tolist()
    if 0 in t:
        t = t[:t.index(0)]
    text = DecodeIds(t).strip("{").strip()
    text = post_process(text.strip())
    print("pred:",text)

def show_input(input, tokenizer):
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    t = input.cpu().numpy()
    t = t.tolist()
    if 0 in t:
        t = t[:t.index(0)]
    text = DecodeIds(t).strip("{").strip()
    # text = post_process(text.strip())
    print("input:",text)

def show_gt(gt, tokenizer):
    gt = post_process(gt.strip())
    print("gt:",gt)

def calc_eval_lines(pred_line, gt, tokenizer):
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                    idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id] or
                    tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    pred_line = pred_line.cpu().tolist()
    gt = gt.cpu().tolist()
    # pred_line = post_process(pred_line.strip())
    # gt = post_process(gt.strip())
    pred_text = DecodeIds(pred_line).strip()
    gt_text = DecodeIds(gt).strip()
    edit_sim = fuzz.ratio(pred_text, gt_text)
    EM = 1 if pred_text == gt_text else 0
    return EM,edit_sim

def post_process(code):
    code = code.replace("<string","<STR_LIT").replace("<number","<NUM_LIT").replace("<char","<CHAR_LIT")
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

if __name__ == '__main__':
    def count_lines(file_path):
        """Return the number of lines in the given file."""
        with open(file_path, 'r') as f:
            return sum(1 for line in f)


    # 使用函数
    file_path = "/data/c/base/RAPT/CodeCompletion/dataset/javaCorpus/token_completion/test.txt"
    # file_path = "/data/c/base/RAPT/CodeCompletion/dataset/py150/token_completion/test.txt"
    # file_path = "/data/c/base/RAPT/CodeCompletion/dataset/py150/token_completion/train.txt"
    line_count = count_lines(file_path)
    print(f"The file {file_path} has {line_count} lines.")
