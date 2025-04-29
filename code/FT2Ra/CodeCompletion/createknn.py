import argparse
import json
import logging
import os
import pickle
import random

import faiss
import numpy as np
import pymongo
import torch
from torch import nn
from tqdm import tqdm


from util import get_model, get_dataloader_token, set_seed, get_online_model, get_dataloader_token_reacc, \
    get_dataloader_token_reacc_train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Copy myself
def knn_llama_relationship(dataname,is_finetuned,save_dir,datadir,epoch=None):
    batch_size = 10
    # modeltype = "llama"
    # pretrained = "../../deepseek/deepseek-coder-6.7b-base"
    # pretrained = "../../codellama/CodeLlama-7b-hf"
    modeltype = "starcoder"
    pretrained = "../../starcoder2/starcoder2-7b"

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned",dataname)

    model, tokenizer, config = get_model(modeltype,pretrained)
    datatype = "train"
    train_dataloader, train_dataset = get_dataloader_token(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir)
    device = torch.device("cuda:0")

    model.to(device)
    # knn_index
    # max_size = len(train_dataset)*1024
    max_size = 5010000
    logger.info("before vecs")
    # vecs = np.zeros((max_size, 4096)).astype('float32')
    vecs = np.zeros((max_size, 4608)).astype('float32')
    logger.info("middle")
    knn_infos = np.zeros(max_size).astype('int32')
    logger.info("after vecs")

    model.eval()
    set_seed(42)
    knn_cnt = 0
    for batch in tqdm(train_dataloader):
        (ids,inputs) = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            valid_lenths = inputs.ne(tokenizer.pad_token_id).sum(-1)
            last_hidden_states = outputs['hidden_states'][-1]
            for (inputs_1d, last_hidden_states_1d, valid_lenth) \
                in zip(inputs, last_hidden_states, valid_lenths):
                inputs_1d = inputs_1d[1:valid_lenth]
                last_hidden_states_1d = last_hidden_states_1d[:valid_lenth-1]
                for position in range(valid_lenth-1):
                    groud_truth = int(inputs_1d[position].item())
                    vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                    knn_infos[knn_cnt] = groud_truth
                    knn_cnt += 1
                    if knn_cnt >= 5000000:
                        break
                if knn_cnt >= 5000000:
                    break
            if knn_cnt >= 5000000:
                break
    print("knn_cnt:{}".format(knn_cnt))
    vecs = vecs[:knn_cnt]
    knn_infos = knn_infos[:knn_cnt]
    # save vecs, knn_info_list to file
    if not os.path.exists(knn_save_dir):
        os.makedirs(knn_save_dir)
    # 保存数据到pickle文件中
    save_file = os.path.join(knn_save_dir, "vecs.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(knn_save_dir, "knn_infos.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(knn_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vecs saved to {knn_save_dir}")
    # knn 部分
    d = len(vecs[1])  # dimension
    nlist = 500  # 量化的聚类中心数，可以调整这个值
    quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
    # train_points_num = 25000
    train_points_num = min(25000, vecs.shape[0])    # update myself
    train_vecs = vecs[np.random.choice(vecs.shape[0], train_points_num, replace=False)]
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 如果没有训练量化器，IVF 索引需要被训练
    assert not index.is_trained
    index.train(train_vecs)
    assert index.is_trained
    # 主数据是 self.vecs
    index.add(vecs)

    # 保存knn_index到文件中
    index_save_path = os.path.join(knn_save_dir, "knn.index")
    faiss.write_index(index, index_save_path)
    print(f"knn.index saved to {knn_save_dir}")


def knn_gpt2_relationship(dataname,is_finetuned,save_dir,datadir,epoch=None):
    batch_size = 10
    modeltype = "gpt2"

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned",dataname)
    # pretrained = "microsoft/CodeGPT-small-java-adaptedGPT2"
    pretrained = "../../codegpt/CodeGPT-small-java-adaptedGPT2"
    if dataname == "py150" or dataname == 'django' or dataname == 'flask' or dataname in [
        "codecov-api", "DocsGPT", "lute-v3", "Python-Type-Challenges", "special-topic-data-engineering", "stable-diffusion-multi-user", "utilmeta-py"]:
        # pretrained = "microsoft/CodeGPT-small-py-adaptedGPT2"
        pretrained = "../../codegpt/CodeGPT-small-py-adaptedGPT2"

    model, tokenizer,config = get_model(modeltype,pretrained)
    datatype = "train"
    train_dataloader, train_dataset = get_dataloader_token(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir)
    device = torch.device("cuda:0")

    model.to(device)
    #knn_index
    # max_size = len(train_dataset)*1024
    max_size = 5010000
    vecs = np.zeros((max_size, 768)).astype('float32')
    knn_infos = np.zeros(max_size).astype('int32')

    model.eval()
    set_seed(42)
    knn_cnt = 0
    for batch in tqdm(train_dataloader):
        (ids,inputs) = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            valid_lenths = inputs.ne(tokenizer.pad_token_id).sum(-1)
            last_hidden_states = outputs['hidden_states'][-1]
            for (inputs_1d, last_hidden_states_1d, valid_lenth) \
                in zip(inputs, last_hidden_states, valid_lenths):
                inputs_1d = inputs_1d[1:valid_lenth]
                last_hidden_states_1d = last_hidden_states_1d[:valid_lenth-1]
                for position in range(valid_lenth-1):
                    groud_truth = int(inputs_1d[position].item())
                    vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                    knn_infos[knn_cnt] = groud_truth
                    knn_cnt += 1
                    if knn_cnt >= 5000000:
                        break
                if knn_cnt >= 5000000:
                    break
            if knn_cnt >= 5000000:
                break
    print("knn_cnt:{}".format(knn_cnt))
    vecs = vecs[:knn_cnt]
    knn_infos = knn_infos[:knn_cnt]
    # save vecs, knn_info_list to file
    if not os.path.exists(knn_save_dir):
        os.makedirs(knn_save_dir)
    # 保存数据到pickle文件中
    save_file = os.path.join(knn_save_dir, "vecs.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(knn_save_dir, "knn_infos.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(knn_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vecs saved to {knn_save_dir}")
    # knn 部分
    d = len(vecs[1])  # dimension
    nlist = 500  # 量化的聚类中心数，可以调整这个值
    quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
    train_points_num = 25000
    train_vecs = vecs[np.random.choice(vecs.shape[0], train_points_num, replace=False)]
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 如果没有训练量化器，IVF 索引需要被训练
    assert not index.is_trained
    index.train(train_vecs)
    assert index.is_trained
    # 主数据是 self.vecs
    index.add(vecs)

    # 保存knn_index到文件中
    index_save_path = os.path.join(knn_save_dir, "knn.index")
    faiss.write_index(index, index_save_path)
    print(f"knn.index saved to {knn_save_dir}")

# Copy myself
def knm_llama_relationship(dataname,is_finetuned,save_dir,datadir,epoch=None):
    batch_size = 10
    # modeltype = "llama"
    # pretrained = "../../deepseek/deepseek-coder-6.7b-base"
    modeltype = 'starcoder'
    pretrained = "../../starcoder2/starcoder2-7b"

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned",dataname)

    model, tokenizer,config = get_model(modeltype,pretrained)
    datatype = "train"
    train_dataloader, train_dataset = get_dataloader_token(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir)
    device = torch.device("cuda:0")
    model.to(device)
    #knn_index
    # max_size = len(train_dataset)*1024
    max_size = 5010000
    # vecs = np.zeros((max_size, 4096)).astype('float32')
    vecs = np.zeros((max_size, 4608)).astype('float32')
    knn_infos = np.zeros(max_size).astype('int32')

    model.eval()
    set_seed(42)
    knn_cnt = 0
    total = 0
    correct = 0
    for batch in tqdm(train_dataloader):
        (ids,inputs) = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            valid_lenths = inputs.ne(tokenizer.pad_token_id).sum(-1)
            last_hidden_states = outputs['hidden_states'][-1]
            logits = outputs['logits']
            for (inputs_1d, last_hidden_states_1d, valid_lenth,logits_1d) \
                in zip(inputs, last_hidden_states, valid_lenths,logits):
                inputs_1d = inputs_1d[1:valid_lenth]
                last_hidden_states_1d = last_hidden_states_1d[:valid_lenth-1]
                for position in range(valid_lenth-1):
                    total += 1
                    if inputs_1d[position] == torch.argmax(logits_1d[position]):
                        correct += 1
                        continue
                    groud_truth = int(inputs_1d[position].item())
                    vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                    knn_infos[knn_cnt] = groud_truth
                    knn_cnt += 1
            if knn_cnt >= 5000000:
                break
    print("knn_cnt:{}".format(knn_cnt))
    wrong_ratio = 1-correct/total
    print("wrong ratio:{}".format(wrong_ratio))
    vecs = vecs[:knn_cnt]
    knn_infos = knn_infos[:knn_cnt]
    # save vecs, knn_info_list to file
    if not os.path.exists(knn_save_dir):
        os.makedirs(knn_save_dir)
    # 保存数据到pickle文件中
    save_file = os.path.join(knn_save_dir, "knm_vecs.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    # save wrong ratio
    save_file = os.path.join(knn_save_dir, "wrong_ratio.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(wrong_ratio, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(knn_save_dir, "knm_knn_infos.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(knn_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vecs saved to {knn_save_dir}")
    # knn 部分
    d = len(vecs[1])  # dimension
    nlist = 500  # 量化的聚类中心数，可以调整这个值
    quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
    # train_points_num = 25000
    train_points_num = min(25000, vecs.shape[0])    # update myself
    train_vecs = vecs[np.random.choice(vecs.shape[0], train_points_num, replace=False)]
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 如果没有训练量化器，IVF 索引需要被训练
    assert not index.is_trained
    index.train(train_vecs)
    assert index.is_trained
    # 主数据是 self.vecs
    index.add(vecs)

    # 保存knn_index到文件中
    index_save_path = os.path.join(knn_save_dir, "knm_knn.index")
    faiss.write_index(index, index_save_path)
    print(f"knn.index saved to {knn_save_dir}")


def knm_gpt2_relationship(dataname,is_finetuned,save_dir,datadir,epoch=None):
    batch_size = 10
    modeltype = "gpt2"

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned",dataname)
    # pretrained = "microsoft/CodeGPT-small-java-adaptedGPT2"
    pretrained = "../../codegpt/CodeGPT-small-java-adaptedGPT2"
    if dataname == "py150" or dataname == 'django' or dataname == 'flask' or dataname in [
        "codecov-api", "DocsGPT", "lute-v3", "Python-Type-Challenges", "special-topic-data-engineering", "stable-diffusion-multi-user", "utilmeta-py"]:
        # pretrained = "microsoft/CodeGPT-small-py-adaptedGPT2"
        pretrained = "../../codegpt/CodeGPT-small-py-adaptedGPT2"

    model, tokenizer,config = get_model(modeltype,pretrained)
    datatype = "train"
    train_dataloader, train_dataset = get_dataloader_token(datadir,modeltype,dataname,datatype,tokenizer,batch_size,save_dir)
    device = torch.device("cuda:0")
    model.to(device)
    #knn_index
    # max_size = len(train_dataset)*1024
    max_size = 5010000
    vecs = np.zeros((max_size, 768)).astype('float32')
    knn_infos = np.zeros(max_size).astype('int32')

    model.eval()
    set_seed(42)
    knn_cnt = 0
    total = 0
    correct = 0
    for batch in tqdm(train_dataloader):
        (ids,inputs) = batch
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            valid_lenths = inputs.ne(tokenizer.pad_token_id).sum(-1)
            last_hidden_states = outputs['hidden_states'][-1]
            logits = outputs['logits']
            for (inputs_1d, last_hidden_states_1d, valid_lenth,logits_1d) \
                in zip(inputs, last_hidden_states, valid_lenths,logits):
                inputs_1d = inputs_1d[1:valid_lenth]
                last_hidden_states_1d = last_hidden_states_1d[:valid_lenth-1]
                for position in range(valid_lenth-1):
                    total += 1
                    if inputs_1d[position] == torch.argmax(logits_1d[position]):
                        correct += 1
                        continue
                    groud_truth = int(inputs_1d[position].item())
                    vecs[knn_cnt] = last_hidden_states_1d[position].cpu().numpy()
                    knn_infos[knn_cnt] = groud_truth
                    knn_cnt += 1
            if knn_cnt >= 5000000:
                break
    print("knn_cnt:{}".format(knn_cnt))
    wrong_ratio = 1-correct/total
    print("wrong ratio:{}".format(wrong_ratio))
    vecs = vecs[:knn_cnt]
    knn_infos = knn_infos[:knn_cnt]
    # save vecs, knn_info_list to file
    if not os.path.exists(knn_save_dir):
        os.makedirs(knn_save_dir)
    # 保存数据到pickle文件中
    save_file = os.path.join(knn_save_dir, "knm_vecs.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
    # save wrong ratio
    save_file = os.path.join(knn_save_dir, "wrong_ratio.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(wrong_ratio, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_file = os.path.join(knn_save_dir, "knm_knn_infos.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(knn_infos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"vecs saved to {knn_save_dir}")
    # knn 部分
    d = len(vecs[1])  # dimension
    nlist = 500  # 量化的聚类中心数，可以调整这个值
    quantizer = faiss.IndexFlatL2(d)  # 使用FlatL2作为量化器
    train_points_num = 25000
    train_vecs = vecs[np.random.choice(vecs.shape[0], train_points_num, replace=False)]
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # 如果没有训练量化器，IVF 索引需要被训练
    assert not index.is_trained
    index.train(train_vecs)
    assert index.is_trained
    # 主数据是 self.vecs
    index.add(vecs)

    # 保存knn_index到文件中
    index_save_path = os.path.join(knn_save_dir, "knm_knn.index")
    faiss.write_index(index, index_save_path)
    print(f"knn.index saved to {knn_save_dir}")


def work_on_completion(save_dir,datadir):
    data_names = [
        # 'sc-django4', 'sc-flask4'
        'sc-spring4', 'sc-android9'
        ]
    for dataname in data_names:
        print("====================================")
        print("data_name:{}".format(dataname))
        # knn_gpt2_relationship(dataname,False,save_dir,datadir)
        # knm_gpt2_relationship(dataname,False,save_dir,datadir)
        knn_llama_relationship(dataname, False, save_dir, datadir)
        knm_llama_relationship(dataname, False, save_dir, datadir)



def main(save_dir, datadir):
    work_on_completion(save_dir, datadir)

    device = torch.device('cuda:0')
    input = 'def sum ( a , b ) :\n'
    
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('../../deepseek/deepseek-coder-6.7b-base', device_map=device)
    tokenizer = AutoTokenizer.from_pretrained('../../deepseek/deepseek-coder-6.7b-base')
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(device)
    while True:
        output = model(input_ids=input_ids,
                       return_dict=True,
                       use_cache=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process save and data directories")
    parser.add_argument('--save_dir', type=str, default="/data/c/base/RAPT/CodeCompletion/save/",
                        help='Directory to save data')
    parser.add_argument('--datadir', type=str, default="/data/c/base/RAPT/CodeCompletion/dataset/",
                        help='Directory with dataset')
    args = parser.parse_args()
    main(args.save_dir, args.datadir)