# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import copy
import os
import pickle
import time

import pandas as pd
import xgboost as xgb
import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F



#  for code completion
#  codegpt，
class CodeCompletionModel(nn.Module):
    def __init__(self, model_class, model_name,config, tokenizer,vecs_path, knn_infos_path,
                 index_save_path,xgboost_path):
        super(CodeCompletionModel, self).__init__()
        # time the model loading
        time1 = time.time()
        self.encoder = model_class.from_pretrained(model_name, config=config)
        self.encoder.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.max_inputs_length = 1024
        time2 = time.time()
        print("load model from {}".format(model_name))
        print("load model cost time:{}".format(time2 - time1))
        with open(vecs_path, 'rb') as f:
            self.vecs = pickle.load(f)
        with open(knn_infos_path, 'rb') as f:
            # train_id, position, groud_truth, pred_id
            self.knn_infos = pickle.load(f)
        self.xgboost = xgb.XGBClassifier()
        if os.path.exists(xgboost_path):
            self.xgboost.load_model(xgboost_path)

        time3 = time.time()
        print("load vecs from {}".format(vecs_path))
        print("load vecs cost time:{}".format(time3 - time2))
        # self.vecs = np.array(self.vecs).astype('float32')
        self.index = faiss.read_index(index_save_path)
        assert self.index.is_trained
        time4 = time.time()
        print("load index from {}".format(index_save_path))
        print("load index cost time:{}".format(time4 - time3))
        # 测试一下knn搜索的时间
        k = 8  # 想要查询的最近邻的数量
        D, I = self.index.search(self.vecs[:100], k)
        print(I[0], D[0])
        time5 = time.time()
        print("test for 100 search items, cost time:{}".format(time5 - time4))

        # add myself
        self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index, faiss.GpuClonerOptions())

        time4 = time.time()
        print("load index from {}".format(index_save_path))
        print("load index cost time:{}".format(time4 - time3))
        # 测试一下knn搜索的时间
        k = 8  # 想要查询的最近邻的数量
        D, I = self.index.search(self.vecs[:100], k)
        print(I[0], D[0])
        time5 = time.time()
        print("test for 100 search items, cost time:{}".format(time5 - time4))

        self.is_knm = "knm" in vecs_path

        # add myself 统计时间
        self.time_ori_generation = 0
        self.time_pt_generation = 0
        self.time_pt_retrieval = 0
        self.time_knm_generation = 0
        self.time_knm_retrieval = 0

    def get_logits_from_vecs(self, vecs):
        with torch.no_grad():
            logits = self.encoder.lm_head(vecs)
        return logits
    
    def get_encoded_vecs_and_logits(self, inputs):
        # 关键函数，gpt2不用改，unixcoder需要改
        with torch.no_grad():
            outputs = self.encoder(inputs, output_hidden_states=True)
            last_hidden_states = outputs['hidden_states'][-1]
            logits = outputs['logits']
        return last_hidden_states, logits

    def get_past_key_values(self,inputs,past_key_values=None):
        with torch.no_grad():
            if past_key_values is None:
                outputs = self.encoder(inputs,output_hidden_states=True)
            else:
                outputs = self.encoder(inputs,past_key_values=past_key_values,output_hidden_states=True)
            past_key_values = outputs.past_key_values
            logits = outputs["logits"]
            last_hidden_states = outputs['hidden_states'][-1]
        return logits,last_hidden_states,past_key_values
    
    def set_device(self, device):
        self.device = device
        self.encoder.to(device)
    
    def set_pt_config(self, max_pt_step, knn_neighbors_num, nprobe, knn_lambda, pt1_lambda, pt2_lambda):
        self.max_pt_step = max_pt_step
        self.knn_neighbors_num = knn_neighbors_num
        self.index.nprobe = nprobe  # 每个查询的聚类中心的数量,越大越准，但是越慢，默认是1
        self.knn_lambda = knn_lambda
        self.pt1_lambda = pt1_lambda
        self.pt2_lambda = pt2_lambda
        self.little_good_w2c_pt = 0
        self.little_good_w2c_knn = 0
        self.all_w2c = 0
        self.total_tokens = 0
        # 当错误变成正确的时候，记录(same_pred_num, self_acc_num, avg_distance,useful_num)
        self.c2w = []
        self.c2c = []
        self.w2c = []
        self.w2w = []

    def set_line_config(self, beam_size, pt_step):
        self.beam_size = beam_size
        self.pt_step = pt_step
    
    def set_knm_lambda(self, knm_lambda):
        self.knm_lambda = knm_lambda

    def get_weights(self,distances, weight_type="weightType1"):
        if weight_type == "weightType1":
            # 使用numpy的广播来计算每个距离的权重
            weights = 1 / (distances + 1)

            # 归一化权重
            weights = weights / weights.sum()

            # return weights
        # if weight_type == "weightType1":
        #     # 倒数权重
        #     def get_weight(distance):
        #         return 1 / (distance + 1)
        #     weight_sum = 0
        #     for i, distance in enumerate(distances):
        #         weight = get_weight(distance)
        #         weight_sum += weight
        #     weightlist = []
        #     for i, distance in enumerate(distances):
        #         weight = get_weight(distance)
        #         weight = weight / weight_sum
        #         weightlist.append(weight)
        #     return weightlist
        elif weight_type == "weightType2":
            n = len(distances)
            weights = np.ones(n) / n  # 创建一个全1的数组并除以其长度
            # return weights
            # # 均匀分布
            # def get_weight(distance):
            #     return 1
            # weight_sum = 0
            # for i, distance in enumerate(distances):
            #     weight = get_weight(distance)
            #     weight_sum += weight
            # weightlist = []
            # for i, distance in enumerate(distances):
            #     weight = get_weight(distance)
            #     weight = weight / weight_sum
            #     weightlist.append(weight)
            # return weightlist
        elif weight_type == "weightType3":
            # distances 是一个 numpy 数组
            distances_tensor = torch.tensor(distances)
            softmax_values = F.softmax(-distances_tensor, dim=-1)
            weights = softmax_values.numpy()
            # return weightlist
        elif weight_type == "weightType4":
            # distances 是一个 numpy 数组
            distances_tensor = torch.pow(torch.tensor(distances),0.5) / torch.tensor(3)
            softmax_values = F.softmax(-distances_tensor, dim=-1)
            weights = softmax_values.numpy()
            # return weightlist
        else:
            raise Exception("weight_type error")
        weights = torch.tensor(weights).to(self.device)
        weights = weights.unsqueeze(-1)
        return weights

    def show_c_w(self,cond_list,name):
        len_list = len(cond_list)
        print("{}: count:{}".format(name, len_list))
        if len_list == 0:
            return
        same_pred_num_avg = sum(x[0] for x in cond_list) / len_list
        same_pred_num_0 = len([x for x in cond_list if x[0] == 0])
        same_pred_num_1 = len([x for x in cond_list if x[0] == 1])
        self_acc_num_avg = sum(x[1] for x in cond_list) / len_list
        self_acc_num_0 = len([x for x in cond_list if x[1] == 0])
        self_acc_num_1 = len([x for x in cond_list if x[1] == 1])
        avg_distance_avg = sum(x[2] for x in cond_list) / len_list
        distance_mt_300 = len([x for x in cond_list if x[2] > 300])
        distance_mt_320 = len([x for x in cond_list if x[2] > 320])
        useful_num_avg = sum(x[3] for x in cond_list) / len_list
        useful_num_0 = len([x for x in cond_list if x[3] == 0])
        useful_num_1 = len([x for x in cond_list if x[3] == 1])
        print("{}: same_pred_num_avg:{}, self_acc_num_avg:{}, avg_distance_avg:{}"
              .format(name,same_pred_num_avg, self_acc_num_avg, avg_distance_avg))
        print("{}: same_pred_num_0:{}, same_pred_num_1:{}, self_acc_num_0:{}, self_acc_num_1:{}"
              .format(name,same_pred_num_0, same_pred_num_1, self_acc_num_0, self_acc_num_1))
        print("{}: useful_num_avg:{}, useful_num_0:{}, useful_num_1:{}"
                .format(name,useful_num_avg, useful_num_0, useful_num_1))
        print("{}: distance_mt_300:{}, distance_mt_320:{}"
                .format(name,distance_mt_300, distance_mt_320))
    def show_0good_w2c(self):
        print("wrong to correct:{:.4f}".format(self.all_w2c/self.total_tokens))
        print("pt little good neighbors wrong to correct:{:.4f}".format(self.little_good_w2c_pt/self.total_tokens))
        print("knn little good neighbors wrong to correct:{:.4f}".format(self.little_good_w2c_knn/self.total_tokens))
        # 查看有多少个预测是通过很少的好邻居也能将错误的修复的
    def predict_tokens_knm(self, inputs):
        assert self.is_knm, "predict_tokens_knm only for knm model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knm的结果存放的数组
            knm_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # knm 部分：
                slide_window = 8
                if position< slide_window+1:
                    knm_lambda = self.knm_lambda
                else:
                    error_num = (inputs[position-slide_window:position] != pred_ids[position-slide_window-1:position-1]).sum()
                    knm_lambda = 0.5*(self.knm_lambda + error_num/slide_window)
                knm_probs = probs_x.clone()
                knm_probs = knm_probs * (1 - knm_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knm_probs += knm_lambda * (weights * neighbor_gts_onehot).sum(0)
                knm_pred_ids[position] = knm_probs.argmax(-1)
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knm_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_" + str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_" + str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_reacc(self, inputs,context_len):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth - 1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                if position < context_len:
                    continue
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position + 1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 观察邻居的特性
                ori_correct = ground_truth_x == pred_ids[position]
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn_pred_ids[position] = knn_probs.argmax(-1)
                knn_correct = ground_truth_x == knn_pred_ids[position]
                # pt2 部分：
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt2_pred_ids_list[step][position] = pt_pred_id
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_" + str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_" + str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens(self, inputs):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 观察邻居的特性
                ori_correct = ground_truth_x == pred_ids[position]
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn_pred_ids[position] = knn_probs.argmax(-1)
                knn_correct = ground_truth_x == knn_pred_ids[position]
                # # pt1 部分：这次论文不提这个了
                # pt1_probs = probs_x.clone()
                # pt1_neighbors = copy.deepcopy(pt_neighbors)
                # for step in range(self.max_pt_step):
                #     # 假想训练，
                #     delta_probs, pt1_neighbors = self.pseudo_train_1(pt1_neighbors, weights, self.pt1_lambda)
                #     pt1_probs += delta_probs
                #     pt_pred_id = pt1_probs.argmax(-1)
                #     # 更新，位置为position
                #     pt1_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt2_pred_ids_list[step][position] = pt_pred_id
                # # for 0 good neighbors wrong to correct
                # pt2_correct = ground_truth_x == pt2_pred_ids_list[self.pt_step][position]
                # usefull_num = (neighbor_gts == ground_truth_x).sum()
                # if not ori_correct and pt2_correct:
                #     self.all_w2c += 1
                #     if usefull_num <= 2:
                #         self.little_good_w2c_pt += 1
                # if not ori_correct and knn_correct and usefull_num <= 2:
                #     self.little_good_w2c_knn += 1
                # self.total_tokens += 1
                # # tune the neighbor rule
                # avg_logits_distance = torch.norm(neighbor_logits-logits_x.unsqueeze(0), dim=1).mean()
                # neighbor_preds = torch.argmax(neighbor_probs, dim=-1)
                # same_pred_num = (neighbor_preds == pred_ids[position]).sum()
                # self_acc_num = (neighbor_preds == neighbor_gts).sum()
                # usefull_num = (neighbor_preds == ground_truth_x).sum()
                # pt2_correct = ground_truth_x == pt2_pred_ids_list[0][position]
                # if ori_correct:
                #     if pt2_correct:
                #         self.c2c.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                #     else:
                #         self.c2w.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                # elif not ori_correct:
                #     if pt2_correct:
                #         self.w2c.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                #     else:
                #         self.w2w.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_rq4_real_train(self, inputs):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with (torch.no_grad()):
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # pt2 部分：
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt1_pred_ids_list[step][position] = pt_pred_id
                # 真训练
                rt_neighbors, weights = self.init_rt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 复制权重和偏置
                lm_head1 = nn.Linear(self.encoder.config.n_embd, self.encoder.config.vocab_size, bias=False).to(self.device)
                lm_head1.weight.data = self.encoder.lm_head.weight.data.clone()
                lm_head2 = nn.Linear(1, self.encoder.config.vocab_size, bias=False).to(self.device)
                lm_head2.weight.data.zero_()
                # rt_lambdas = [self.pt2_lambda*1.1]+[self.pt2_lambda*0.2] +\
                #              [self.pt2_lambda*0.1]+[self.pt2_lambda*0.1] +\
                #              [self.pt2_lambda*0.05]+[self.pt2_lambda*0.05] +\
                #              [self.pt2_lambda*0.01]*(self.max_pt_step-6)
                rt_lambdas = [self.pt2_lambda]*self.max_pt_step
                for step in range(self.max_pt_step):
                    with torch.enable_grad():
                        lm_head1 = self.real_train1(rt_neighbors, weights, rt_lambdas[step], lm_head1)
                        # lm_head1 = self.real_train1_noweight(rt_neighbors, weights, self.pt2_lambda/500, lm_head1)
                    logits_tmp = lm_head1(vec_x)
                    pred_id = logits_tmp.argmax(-1)
                    pt2_pred_ids_list[step][position] = pred_id

                    # with torch.enable_grad():
                    #     lm_head2 = self.real_train2(rt_neighbors, weights, self.pt2_lambda*1.2, lm_head2)
                    # vector = torch.ones(1).to(self.device)
                    # logits_tmp = lm_head2(vector)
                    # logits_tmp += logits_x
                    # pred_id = logits_tmp.argmax(-1)
                    # pt2_pred_ids_list[step][position] = pred_id
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def real_train2(self,rt_neighbors, weights, lr, lm_head):
        # 真实训练 lm_head，目标是残差
        # lm_head 形状是 (1,vocab_size)
        if lm_head.weight.grad is not None:
            lm_head.weight.grad.data.zero_()
        neighbors_vecs, neighbors_logits, neighbor_gts = rt_neighbors
        vector = torch.ones(neighbors_vecs.shape[0], 1).to(self.device)
        residual_logits = lm_head(vector)
        neighbors_logits_new = neighbors_logits.detach() + residual_logits

        # # 计算每个样本的交叉熵损失
        # losses = F.cross_entropy(neighbors_logits_new, neighbor_gts, reduction='none')
        # # 应用样本权重
        # weighted_losses = losses * weights
        # loss = weighted_losses.sum()
        # loss.backward()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(neighbors_logits_new, neighbor_gts)
        loss.backward()
        lm_head.weight.data -= lr * lm_head.weight.grad.data

        return lm_head
    def real_train2_no(self,rt_neighbors, weights, lr, lm_head):
        # 真实训练 lm_head，目标是残差
        # lm_head 形状是 (1,vocab_size)
        if lm_head.weight.grad is not None:
            lm_head.weight.grad.data.zero_()
        neighbors_vecs, neighbors_logits, neighbor_gts = rt_neighbors
        vector = torch.ones(neighbors_vecs.shape[0], 1).to(self.device)
        residual_logits = lm_head(vector)
        neighbors_logits_new = neighbors_logits.detach() + residual_logits

        # 计算每个样本的交叉熵损失
        losses = F.cross_entropy(neighbors_logits_new, neighbor_gts, reduction='none')
        # 应用样本权重
        weighted_losses = losses * weights
        loss = weighted_losses.sum()
        loss.backward()
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(neighbors_logits, neighbor_gts)
        # loss.backward()
        lm_head.weight.data -= lr * lm_head.weight.grad.data
        with torch.no_grad():
            residual_logits = lm_head(vector)
            neighbors_logits_new = neighbors_logits.detach() + residual_logits
            rt_neighbors = (neighbors_vecs, neighbors_logits_new, neighbor_gts)
        return lm_head,rt_neighbors
    def real_train1(self,rt_neighbors, weights, lr, lm_head):
        # 真实训练 lm_head,完全打不过假训练
        # lm_head 形状是 (n_embd,vocab_size)
        if lm_head.weight.grad is not None:
            lm_head.weight.grad.data.zero_()
        neighbors_vecs, neighbors_logits_old, neighbor_gts = rt_neighbors
        neighbors_logits = lm_head(neighbors_vecs)
        # 方案1
        # 计算每个样本的交叉熵损失
        losses = F.cross_entropy(neighbors_logits, neighbor_gts, reduction='none')
        # 应用样本权重
        weighted_losses = losses * weights
        loss = weighted_losses.sum()
        loss.backward()
        # 方案2
        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(neighbors_logits, neighbor_gts)
        # loss.backward()
        lm_head.weight.data -= lr * lm_head.weight.grad.data
        return lm_head
    def real_train1_noweight(self,rt_neighbors, weights, lr, lm_head):
        # 真实训练 lm_head,完全打不过假训练
        # lm_head 形状是 (n_embd,vocab_size)
        if lm_head.weight.grad is not None:
            lm_head.weight.grad.data.zero_()
        neighbors_vecs, neighbors_logits_old, neighbor_gts = rt_neighbors
        neighbors_logits = lm_head(neighbors_vecs)
        # 方案2
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(neighbors_logits, neighbor_gts)
        loss.backward()
        lm_head.weight.data -= lr * lm_head.weight.grad.data
        return lm_head
    def predict_tokens_rq2_epoch(self, inputs):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 观察邻居的特性
                ori_correct = ground_truth_x == pred_ids[position]
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn_pred_ids[position] = knn_probs.argmax(-1)
                # pt2 single epoch 部分：
                pt1_logits_single = logits_x.clone()
                pt1_neighbors_single = copy.deepcopy(pt_neighbors)
                delta_logits, pt1_neighbors_single = self.pseudo_train_2(pt1_neighbors_single, weights, self.pt1_lambda)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    pt1_logits_single += delta_logits
                    pt_pred_id = pt1_logits_single.argmax(-1)
                    # 更新，位置为position
                    # 存到pt1_pred_ids_list里面
                    pt1_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt2_pred_ids_list[step][position] = pt_pred_id
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_rq2_lambda(self, inputs):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            pt3_pred_ids_list = []
            pt4_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
                pt3_pred_ids_list.append(pred_ids.clone())
                pt4_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # pt2 部分：
                pt2_lambda = self.pt2_lambda*0.5
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt1_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_lambda = self.pt2_lambda
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt2_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_lambda = self.pt2_lambda * 2
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt3_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_lambda = self.pt2_lambda * 4
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt4_pred_ids_list[step][position] = pt_pred_id
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
                pred_ids_dict["pt3_"+str(step)] = pt3_pred_ids_list[step]
                pred_ids_dict["pt4_"+str(step)] = pt4_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_rq3_weight(self, inputs):
        # assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            # knn_pred_ids = pred_ids.clone()
            knn1_pred_ids = pred_ids.clone()
            knn2_pred_ids = pred_ids.clone()
            knn3_pred_ids = pred_ids.clone()
            knn4_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            pt3_pred_ids_list = []
            pt4_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
                pt3_pred_ids_list.append(pred_ids.clone())
                pt4_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 更换weights再算一次
                weights = self.get_weights(distances, weight_type="weightType1")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt1_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn1_pred_ids[position] = knn_probs.argmax(-1)
                # 更换weights再算一次
                weights = self.get_weights(distances, weight_type="weightType2")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt2_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn2_pred_ids[position] = knn_probs.argmax(-1)
                # 更换weights再算一次
                weights = self.get_weights(distances, weight_type="weightType3")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt3_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn3_pred_ids[position] = knn_probs.argmax(-1)
                # 更换weights再算一次
                weights = self.get_weights(distances, weight_type="weightType4")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt4_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn4_pred_ids[position] = knn_probs.argmax(-1)
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn1"] = knn1_pred_ids
            pred_ids_dict["knn2"] = knn2_pred_ids
            pred_ids_dict["knn3"] = knn3_pred_ids
            pred_ids_dict["knn4"] = knn4_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
                pred_ids_dict["pt3_"+str(step)] = pt3_pred_ids_list[step]
                pred_ids_dict["pt4_"+str(step)] = pt4_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_rq3_loss(self, inputs):
        assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            knn_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # 观察邻居的特性
                ori_correct = ground_truth_x == pred_ids[position]
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn_pred_ids[position] = knn_probs.argmax(-1)
                knn_correct = ground_truth_x == knn_pred_ids[position]
                # # pt1 部分：这次论文不提这个了
                # pt1_probs = probs_x.clone()
                # pt1_neighbors = copy.deepcopy(pt_neighbors)
                # for step in range(self.max_pt_step):
                #     # 假想训练，
                #     delta_probs, pt1_neighbors = self.pseudo_train_1(pt1_neighbors, weights, self.pt1_lambda)
                #     pt1_probs += delta_probs
                #     pt_pred_id = pt1_probs.argmax(-1)
                #     # 更新，位置为position
                #     pt1_pred_ids_list[step][position] = pt_pred_id
                # pt2 部分：
                pt2_logits = logits_x.clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.max_pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                    pt_pred_id = pt2_logits.argmax(-1)
                    # 更新，位置为position
                    pt2_pred_ids_list[step][position] = pt_pred_id
                # # for 0 good neighbors wrong to correct
                # pt2_correct = ground_truth_x == pt2_pred_ids_list[self.pt_step][position]
                # usefull_num = (neighbor_gts == ground_truth_x).sum()
                # if not ori_correct and pt2_correct:
                #     self.all_w2c += 1
                #     if usefull_num <= 2:
                #         self.little_good_w2c_pt += 1
                # if not ori_correct and knn_correct and usefull_num <= 2:
                #     self.little_good_w2c_knn += 1
                # self.total_tokens += 1
                # # tune the neighbor rule
                # avg_logits_distance = torch.norm(neighbor_logits-logits_x.unsqueeze(0), dim=1).mean()
                # neighbor_preds = torch.argmax(neighbor_probs, dim=-1)
                # same_pred_num = (neighbor_preds == pred_ids[position]).sum()
                # self_acc_num = (neighbor_preds == neighbor_gts).sum()
                # usefull_num = (neighbor_preds == ground_truth_x).sum()
                # pt2_correct = ground_truth_x == pt2_pred_ids_list[0][position]
                # if ori_correct:
                #     if pt2_correct:
                #         self.c2c.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                #     else:
                #         self.c2w.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                # elif not ori_correct:
                #     if pt2_correct:
                #         self.w2c.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
                #     else:
                #         self.w2w.append((same_pred_num, self_acc_num, avg_logits_distance,usefull_num))
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn"] = knn_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
        return pred_ids_dict
    def predict_tokens_rq3_neighbornum(self, inputs):
        # assert not self.is_knm, "predict_tokens only for knn model"
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)

            # 准备knn和knn-pt的结果存放的数组
            # knn_pred_ids = pred_ids.clone()
            knn1_pred_ids = pred_ids.clone()
            knn2_pred_ids = pred_ids.clone()
            knn3_pred_ids = pred_ids.clone()
            knn4_pred_ids = pred_ids.clone()
            pt1_pred_ids_list = []
            pt2_pred_ids_list = []
            pt3_pred_ids_list = []
            pt4_pred_ids_list = []
            for step in range(self.max_pt_step):
                pt1_pred_ids_list.append(pred_ids.clone())
                pt2_pred_ids_list.append(pred_ids.clone())
                pt3_pred_ids_list.append(pred_ids.clone())
                pt4_pred_ids_list.append(pred_ids.clone())
            # 批量knn搜索
            vec_search = vecs[:valid_lenth-1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            for position in range(valid_lenth - 1):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                ground_truth_x = inputs[position+1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                assert self.knn_neighbors_num == 50, "knn_neighbors_num should be 50 in this experiment"
                # 更换weights再算一次
                neighbors_num = 5
                weights = self.get_weights(distances[:neighbors_num], weight_type="weightType1")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    logits_nb, probs_nb, gts_nb, gts_onehot_nb = pt2_neighbors
                    pt2_neighbors = (logits_nb[:neighbors_num], probs_nb[:neighbors_num], gts_nb[:neighbors_num], gts_onehot_nb[:neighbors_num])
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt1_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                neighbor_gts_onehot = neighbor_gts_onehot[:neighbors_num]
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn1_pred_ids[position] = knn_probs.argmax(-1)

                # 更换weights再算一次
                neighbors_num = 10
                weights = self.get_weights(distances[:neighbors_num], weight_type="weightType1")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    logits_nb, probs_nb, gts_nb, gts_onehot_nb = pt2_neighbors
                    pt2_neighbors = (logits_nb[:neighbors_num], probs_nb[:neighbors_num], gts_nb[:neighbors_num],
                                     gts_onehot_nb[:neighbors_num])
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt2_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                neighbor_gts_onehot = neighbor_gts_onehot[:neighbors_num]
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn2_pred_ids[position] = knn_probs.argmax(-1)
                # 更换weights再算一次
                neighbors_num = 20
                weights = self.get_weights(distances[:neighbors_num], weight_type="weightType1")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    logits_nb, probs_nb, gts_nb, gts_onehot_nb = pt2_neighbors
                    pt2_neighbors = (logits_nb[:neighbors_num], probs_nb[:neighbors_num], gts_nb[:neighbors_num],
                                     gts_onehot_nb[:neighbors_num])
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt3_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                neighbor_gts_onehot = neighbor_gts_onehot[:neighbors_num]
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn3_pred_ids[position] = knn_probs.argmax(-1)
                # 更换weights再算一次
                neighbors_num = 50
                weights = self.get_weights(distances[:neighbors_num], weight_type="weightType1")
                if not self.is_knm:
                    # pt2 测试weights部分：
                    pt2_logits = logits_x.clone()
                    pt2_neighbors = copy.deepcopy(pt_neighbors)
                    logits_nb, probs_nb, gts_nb, gts_onehot_nb = pt2_neighbors
                    pt2_neighbors = (logits_nb[:neighbors_num], probs_nb[:neighbors_num], gts_nb[:neighbors_num],
                                     gts_onehot_nb[:neighbors_num])
                    for step in range(self.max_pt_step):
                        # 假想训练，
                        delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                        pt2_logits += delta_logits
                        pt_pred_id = pt2_logits.argmax(-1)
                        # 更新，位置为position
                        pt4_pred_ids_list[step][position] = pt_pred_id
                # knn 部分：
                knn_probs = probs_x.clone()
                knn_probs = knn_probs * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                neighbor_gts_onehot = neighbor_gts_onehot[:neighbors_num]
                knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                knn4_pred_ids[position] = knn_probs.argmax(-1)
            pred_ids_dict = {}
            pred_ids_dict["origin"] = pred_ids
            pred_ids_dict["knn1"] = knn1_pred_ids
            pred_ids_dict["knn2"] = knn2_pred_ids
            pred_ids_dict["knn3"] = knn3_pred_ids
            pred_ids_dict["knn4"] = knn4_pred_ids
            for step in range(self.max_pt_step):
                pred_ids_dict["pt1_"+str(step)] = pt1_pred_ids_list[step]
                pred_ids_dict["pt2_"+str(step)] = pt2_pred_ids_list[step]
                pred_ids_dict["pt3_"+str(step)] = pt3_pred_ids_list[step]
                pred_ids_dict["pt4_"+str(step)] = pt4_pred_ids_list[step]
        return pred_ids_dict
    
    def predict_line_ori(self,inputs,beam_size,break_ids):

        # outputs = self.encoder(inputs[:, :-1])[1]
        logits,last_hidden_states,past_key_values = self.get_past_key_values(inputs[:, :-1])

        # logits, last_hidden_states, past_key_values = self.get_past_key_values(inputs)
        zero = torch.LongTensor(1).fill_(0).to(self.device)
        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                past_key_values]
        past_hidden = [x[:, :1].expand(-1, beam_size, -1, -1, -1) for x in past]
        beam_ori = Beam(beam_size, inputs[0][-1].cpu().data, break_ids, self.device)

        # update myself, 原代码使用的是LogSoftmax
        softmax = torch.nn.Softmax(dim=-1)
        # softmax = torch.nn.LogSoftmax(dim=-1)
        

        for position in range(100):
            if beam_ori.done():
                break
            input_ids = beam_ori.getCurrentState()

            logits, last_hidden_states, past_key_values = self.get_past_key_values(input_ids,past_key_values=past_hidden)

            # outputs = self.encoder(input_ids, past_key_values=past_hidden) # outputs["logits"]
            out = softmax(logits[:, -1, :]).data
            beam_ori.advance(out)
            past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                    past_key_values]
            past_hidden = [x.data.index_select(1, beam_ori.getCurrentOrigin()) for x in past]
        hyp = beam_ori.getHyp(beam_ori.getFinal())
        pred = beam_ori.buildTargetTokens(hyp)[:beam_size]

        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
        pred = torch.cat(pred, 0).unsqueeze(0)
        return pred[0][0]

    """
    kNM-LM补全入口
    """
    def predict_line_knn(self, inputs, beam_size, break_ids):
        # outputs = self.encoder(inputs[:, :-1])[1]
        # logits, last_hidden_states, past_key_values = self.get_past_key_values(inputs)
        t = time.perf_counter()     # add myself
        logits, last_hidden_states, past_key_values = self.get_past_key_values(inputs[:, :-1])
        zero = torch.LongTensor(1).fill_(0).to(self.device)
        # past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
        #         outputs]
        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                past_key_values]
        past_hidden = [x[:, :1].expand(-1, beam_size, -1, -1, -1) for x in past]
        beam_ori = Beam(beam_size, inputs[0][-1].cpu().data, break_ids, self.device)
        softmax = torch.nn.Softmax(dim=-1)
        # softmax = torch.nn.LogSoftmax(dim=-1)
        self.time_knm_generation += time.perf_counter() - t

        for position in range(100):
            if beam_ori.done():
                break

            t = time.perf_counter() # add myself
            input_ids = beam_ori.getCurrentState()
            logits, last_hidden_states, past_key_values = self.get_past_key_values(input_ids,
                                                                                   past_key_values=past_hidden)
            # outputs = self.encoder(input_ids, past_key_values=past_hidden) # outputs["logits"]
            logits_x = logits[:, 0, :]
            probs_x = softmax(logits_x).data
            vec_x = last_hidden_states[:, 0, :]
            self.time_knm_generation += time.perf_counter() - t # add myself

            t = time.perf_counter() # add myself
            # knn搜索邻居部分
            # 批量knn搜索
            vec_search = vec_x.cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            knn_probs = probs_x.clone()
            for beam in range(beam_size):
                distances = D[beam]
                neighbors_id = I[beam]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # weights = self.get_weights(distances, weight_type="weightType4")
                # knn 部分：
                knn_probs[beam] = knn_probs[beam] * (1 - self.knn_lambda)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                knn_probs[beam] += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
                # self.tokenizer.decode(knn_probs[0].argmax(-1).cpu().data.numpy())
                # knn_probs[0].argmax(-1)
                # top3_indices = knn_probs[0].argsort()[::-1][:3]
                # top3_indices = (weights * neighbor_gts_onehot).sum(0).argsort()[::-1][:3]
            # 更新knn_probs，得到log(knn_probs)
            knn_probs = torch.log(knn_probs)
            beam_ori.advance(knn_probs)
            past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                    past_key_values]
            past_hidden = [x.data.index_select(1, beam_ori.getCurrentOrigin()) for x in past]
            self.time_knm_retrieval += time.perf_counter() - t  # add myself

        hyp = beam_ori.getHyp(beam_ori.getFinal())
        pred = beam_ori.buildTargetTokens(hyp)[:beam_size]

        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
        pred = torch.cat(pred, 0).unsqueeze(0)
        return pred[0][0]
    
    """
    FT2Ra补全入口
    """
    def predict_line_pt(self, inputs, beam_size, break_ids):
        # outputs = self.encoder(inputs[:, :-1])[1]
        t = time.perf_counter()     # add myself
        logits, last_hidden_states, past_key_values = self.get_past_key_values(inputs[:, :-1])
        # logits, last_hidden_states, past_key_values = self.get_past_key_values(inputs)
        zero = torch.LongTensor(1).fill_(0).to(self.device)
        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                past_key_values]
        past_hidden = [x[:, :1].expand(-1, beam_size, -1, -1, -1) for x in past]
        beam_ori = Beam(beam_size, inputs[0][-1].cpu().data, break_ids, self.device)
        # softmax = torch.nn.Softmax(dim=-1)
        softmax = torch.nn.LogSoftmax(dim=-1)
        self.time_pt_generation += time.perf_counter() - t      # add myself
        
        for position in range(100):
            if beam_ori.done():
                break

            t = time.perf_counter() # add myself
            input_ids = beam_ori.getCurrentState()
            # outputs = self.encoder(input_ids, past_key_values=past_hidden,output_hidden_states=True)
            logits, last_hidden_states, past_key_values = self.get_past_key_values(input_ids,
                                                                                   past_key_values=past_hidden)
            logits_x = logits[:, 0, :]
            probs_x = softmax(logits_x).data
            vec_x = last_hidden_states[:, 0, :]
            self.time_pt_generation += time.perf_counter() - t  # add myself

            t = time.perf_counter() # add myself
            # knn搜索邻居部分
            # 批量knn搜索
            vec_search = vec_x.cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            pt_probs = probs_x.clone()
            for beam in range(beam_size):
                distances = D[beam]
                neighbors_id = I[beam]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # pt2 部分：
                pt2_logits = logits_x[beam].clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                pt_probs[beam] = softmax(pt2_logits)
            beam_ori.advance(pt_probs)
            past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                    past_key_values]
            # past = [torch.cat([x[0][:,:,:current_postion,:].unsqueeze(0), x[1][:,:,:current_postion,:].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in outputs[1]]
            past_hidden = [x.data.index_select(1, beam_ori.getCurrentOrigin()) for x in past]
            self.time_pt_retrieval += time.perf_counter() - t   # add myself

        hyp = beam_ori.getHyp(beam_ori.getFinal())
        pred = beam_ori.buildTargetTokens(hyp)[:beam_size]

        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
        pred = torch.cat(pred, 0).unsqueeze(0)
        return pred[0][0]
    def predict_line_pt_backup(self, inputs, beam_size, break_ids):
        outputs = self.encoder(inputs[:, :-1])[1]
        zero = torch.LongTensor(1).fill_(0).to(self.device)
        past = [torch.cat([x[0].unsqueeze(0), x[1].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in
                outputs]
        past_hidden = [x[:, :1].expand(-1, beam_size, -1, -1, -1) for x in past]
        beam_ori = Beam(beam_size, inputs[0][-1].cpu().data, break_ids, self.device)
        softmax = torch.nn.LogSoftmax(dim=-1)
        for position in range(100):
            if beam_ori.done():
                break
            input_ids = beam_ori.getCurrentState()
            current_postion = inputs.shape[1] + position
            # pad = torch.full((input_ids.shape[0], self.max_inputs_length - current_postion - 1),
            #                  self.tokenizer.pad_token_id).to(self.device)
            # input_ids = torch.cat((input_ids, pad), -1)
            outputs = self.encoder(input_ids, past_key_values=past_hidden,output_hidden_states=True)
            logits_x = outputs["logits"][:, 0, :]
            probs_x = softmax(logits_x).data
            vec_x = outputs['hidden_states'][-1][:, 0, :]
            # knn搜索邻居部分
            # 批量knn搜索
            vec_search = vec_x.cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            pt_probs = probs_x.clone()
            for beam in range(beam_size):
                distances = D[beam]
                neighbors_id = I[beam]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                # pt2 部分：
                pt2_logits = logits_x[beam].clone()
                pt2_neighbors = copy.deepcopy(pt_neighbors)
                for step in range(self.pt_step):
                    # 假想训练，
                    delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                    pt2_logits += delta_logits
                pt_probs[beam] = softmax(pt2_logits)
            beam_ori.advance(pt_probs)
            past = []
            for x in outputs[1]:
                if type(x) == tuple:
                    modified_x = torch.cat([x[0][:, :, :current_postion, :].unsqueeze(0),
                                            x[1][:, :, :current_postion, :].unsqueeze(0)], dim=0)
                    past.append(modified_x)
                else:
                    past.append(x[:, :, :current_postion, :])
            # past = [torch.cat([x[0][:,:,:current_postion,:].unsqueeze(0), x[1][:,:,:current_postion,:].unsqueeze(0)], dim=0) if type(x) == tuple else x for x in outputs[1]]
            past_hidden = [x.data.index_select(1, beam_ori.getCurrentOrigin()) for x in past]
        hyp = beam_ori.getHyp(beam_ori.getFinal())
        pred = beam_ori.buildTargetTokens(hyp)[:beam_size]

        pred = [torch.cat([x.view(-1) for x in p] + [zero] * (100 - len(p))).view(1, -1) for p in pred]
        pred = torch.cat(pred, 0).unsqueeze(0)
        return pred[0][0]
    
    """
    补全代码的方法入口
    """
    def predict_line_new(self,inputs,break_ids,knn_need=False,pt_need=False):
        # 断言，input_ids是2维的,但是只有一行
        assert len(inputs.shape) == 2, "input_ids should be 2-dim"
        assert inputs.shape[0] == 1, "input_ids should only have one line"
        # beam_size = 5
        beam_size = 1   # update myself
        # t = time.perf_counter()
        # pred_line_ori = self.predict_line_ori(inputs,beam_size,break_ids)
        # self.time_ori_generation += time.perf_counter() - t

        # pred_line_ori = inputs[0][:1]
        if knn_need:
            pred_line_knn = self.predict_line_knn(inputs,beam_size,break_ids)
        # else:
        #     pred_line_knn = pred_line_ori
        if pt_need:
            pred_line_pt = self.predict_line_pt(inputs,beam_size,break_ids)
        # else:
        #     pred_line_pt = pred_line_ori


        pred_line_dict = {}
        # pred_line_dict["origin"] = pred_line_ori
        pred_line_dict["knn"] = pred_line_knn
        # pred_line_dict["pt"] = pred_line_pt
        return pred_line_dict
    def predict_line(self,inputs,max_len,gt):
        raise RuntimeError("predict_line is deprecated, use predict_line_new instead")
        # 断言，input_ids是1维的,但是只有一行
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        # 扩展成beam_size个
        inputs_ori = inputs.repeat(self.beam_size, 1)
        inputs_knn = inputs.repeat(self.beam_size, 1)
        inputs_pt = inputs.repeat(self.beam_size, 1)
        eos_id = self.tokenizer.eos_token_id
        beam_ori = Beam(self.beam_size, eos_id, self.device)
        beam_knn = Beam(self.beam_size, eos_id, self.device)
        beam_pt = Beam(self.beam_size, eos_id, self.device)
        for position in range(max_len):
            pre_logits_ori = self.predict_one_token_ori(inputs_ori)
            pre_logits_knn = self.predict_one_token_knn(inputs_knn)
            pre_logits_pt = self.predict_one_token_pt(inputs_pt)
            beam_ori.advance(pre_logits_ori)
            beam_knn.advance(pre_logits_knn)
            beam_pt.advance(pre_logits_pt)
            inputs_ori.data.copy_(inputs_ori.data.index_select(0, beam_ori.getCurrentOrigin()))
            inputs_knn.data.copy_(inputs_knn.data.index_select(0, beam_knn.getCurrentOrigin()))
            inputs_pt.data.copy_(inputs_pt.data.index_select(0, beam_pt.getCurrentOrigin()))
            inputs_ori = torch.cat((inputs_ori, beam_ori.getCurrentState().clone()), -1)
            inputs_knn = torch.cat((inputs_knn, beam_knn.getCurrentState().clone()), -1)
            inputs_pt = torch.cat((inputs_pt, beam_pt.getCurrentState().clone()), -1)
            # if beam.done():
            #     break
            # 不退出了，一直预测到max_len
        # Extract the sequences from the beam
        pred_line_dict = {}
        pred_line_dict["origin"] = inputs_ori[0][-max_len:]
        pred_line_dict["knn"] = inputs_knn[0][-max_len:]
        pred_line_dict["pt"] = inputs_pt[0][-max_len:]
        return pred_line_dict
    def pad_to_fixed_length(self, inputs):
        if len(inputs.shape)==1:
            pad = torch.full((self.max_inputs_length - len(inputs) - 1,), self.tokenizer.pad_token_id).to(self.device)
        else:
            pad = torch.full((inputs.shape[0],self.max_inputs_length - inputs.shape[1] - 1), self.tokenizer.pad_token_id).to(self.device)
        inputs = torch.cat((inputs, pad), -1)
        return inputs
    def get_logits_with_past(self,inputs,past_hidden):
        pass

    def predict_one_token_ori_with_past(self,inputs,past_hidden):
        key_position = inputs.shape[1] - 1
        inputs = self.pad_to_fixed_length(inputs)
        logits = self.get_logits_with_past(inputs,past_hidden)
        logits = logits[:, key_position, :]
        probs = F.softmax(logits, dim=-1)
        return probs
    def predict_one_token_ori(self,inputs):
        key_position = inputs.shape[1] - 1
        # inputs = self.pad_to_fixed_length(inputs)
        vecs, logits = self.get_encoded_vecs_and_logits(inputs)
        vecs = vecs[:, key_position, :]
        logits = logits[:, key_position, :]
        probs = F.softmax(logits, dim=-1)
        return probs
    def predict_one_token_knn(self,inputs):
        key_position = inputs.shape[1] - 1
        inputs = self.pad_to_fixed_length(inputs)
        vecs, logits = self.get_encoded_vecs_and_logits(inputs)
        vecs = vecs[:, key_position, :]
        logits = logits[:, key_position, :]
        probs = F.softmax(logits, dim=-1)
        vecs = vecs.cpu().detach().numpy()
        # knn搜索邻居
        D, I = self.index.search(vecs, self.knn_neighbors_num)
        # knn 部分：
        knn_probs_beam = []
        for i in range(inputs.shape[0]):
            vec_x = vecs[i]
            logits_x = logits[i]
            probs_x = probs[i]
            distances = D[i]
            neighbors_id = I[i]
            pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
            # knn 部分：
            knn_probs = probs_x.clone()
            knn_probs = knn_probs * (1 - self.knn_lambda)
            neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
            knn_probs += self.knn_lambda * (weights * neighbor_gts_onehot).sum(0)
            knn_probs_beam.append(knn_probs)
        knn_probs_beam = torch.stack(knn_probs_beam).to(self.device)
        return knn_probs_beam
    def predict_one_token_pt(self,inputs):
        key_position = inputs.shape[1] - 1
        inputs = self.pad_to_fixed_length(inputs)
        vecs, logits = self.get_encoded_vecs_and_logits(inputs)
        vecs = vecs[:, key_position, :]
        logits = logits[:, key_position, :]
        probs = F.softmax(logits, dim=-1)
        vecs = vecs.cpu().detach().numpy()
        # knn搜索邻居
        D, I = self.index.search(vecs, self.knn_neighbors_num)
        # knn-pt 部分：
        pt_probs_beam = []
        for i in range(inputs.shape[0]):
            vec_x = vecs[i]
            logits_x = logits[i]
            probs_x = probs[i]
            distances = D[i]
            neighbors_id = I[i]
            pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
            # # pt1 部分：
            # pt1_probs = probs_x.clone()
            # pt1_neighbors = copy.deepcopy(pt_neighbors)
            # for step in range(self.max_pt_step):
            #     # 假想训练，
            #     delta_probs, pt1_neighbors = self.pseudo_train_1(pt1_neighbors, weights, self.pt1_lambda)
            #     pt1_probs += delta_probs
            # pt_probs_beam.append(pt1_probs)
            # pt2 部分：
            pt2_logits = logits_x.clone()
            pt2_neighbors = copy.deepcopy(pt_neighbors)
            for step in range(self.pt_step):
                # 假想训练，
                delta_logits, pt2_neighbors = self.pseudo_train_2(pt2_neighbors, weights, self.pt2_lambda)
                pt2_logits += delta_logits
            pt2_probs = F.softmax(pt2_logits, dim=-1)
            pt_probs_beam.append(pt2_probs)
        pt_probs_beam = torch.stack(pt_probs_beam).to(self.device)
        return pt_probs_beam
    def init_rt_neighbors(self, vec_x, logits_x, probs_x, distances, neighbor_ids):
        # neighbors_vecs,neighbor_gts
        neighbors_vecs = self.vecs[neighbor_ids]
        neighbors_vecs = torch.tensor(neighbors_vecs).to(self.device)
        neighbors_logits = self.get_logits_from_vecs(neighbors_vecs)
        # neighbors_probs = F.softmax(neighbors_logits, dim=-1)
        neighbor_gts = self.knn_infos[neighbor_ids]
        neighbor_gts = torch.tensor(neighbor_gts, dtype=torch.long).to(self.device)
        rt_neighbors = (neighbors_vecs[:self.knn_neighbors_num],
                        neighbors_logits[:self.knn_neighbors_num],
                        neighbor_gts[:self.knn_neighbors_num])
        # 直接用distance计算权重
        weights = self.get_weights(distances)
        return rt_neighbors, weights
    def init_pt_neighbors(self, vec_x, logits_x, probs_x, distances, neighbor_ids):
        # vec_i, logits_i, ground_truth_i, sf_logits_i
        neighbors_vecs = self.vecs[neighbor_ids]
        # convert to torch.tensor
        neighbors_vecs = torch.tensor(neighbors_vecs).to(self.device)
        neighbors_logits = self.get_logits_from_vecs(neighbors_vecs)
        neighbors_probs = F.softmax(neighbors_logits, dim=-1)
        neighbor_gts = self.knn_infos[neighbor_ids]
        neighbor_gts_onehot = torch.zeros_like(neighbors_probs)
        for i in range(len(neighbor_gts)):
            neighbor_gts_onehot[i][neighbor_gts[i]] = 1
        neighbor_gts = torch.tensor(neighbor_gts).to(self.device)
        pt_neighbors = (neighbors_logits[:self.knn_neighbors_num],
                        neighbors_probs[:self.knn_neighbors_num],
                        neighbor_gts[:self.knn_neighbors_num],
                        neighbor_gts_onehot[:self.knn_neighbors_num])
        # 直接用distance计算权重
        weights = self.get_weights(distances)
        # weights = torch.tensor(weights).to(self.device)
        # weights = weights.unsqueeze(-1)
        return pt_neighbors, weights
    def init_pt_neighbors_new(self, vec_x, logits_x,probs_x, distances, neighbor_ids):
        raise Exception("not implement")
        # vec_i, logits_i, ground_truth_i, sf_logits_i
        neighbors_vecs = self.vecs[neighbor_ids]
        # convert to torch.tensor
        neighbors_vecs = torch.tensor(neighbors_vecs).to(self.device)
        neighbors_logits = self.get_logits_from_vecs(neighbors_vecs)
        neighbors_probs = F.softmax(neighbors_logits, dim=-1)

        neighbor_gts = self.knn_infos[neighbor_ids]
        neighbor_gts_onehot = torch.zeros_like(neighbors_probs)
        for i in range(len(neighbor_gts)):
            neighbor_gts_onehot[i][neighbor_gts[i]] = 1
        neighbor_gts = torch.tensor(neighbor_gts).to(self.device)
        pt_neighbors = (neighbors_logits[:self.knn_neighbors_num],
                        neighbors_probs[:self.knn_neighbors_num],
                        neighbor_gts[:self.knn_neighbors_num],
                        neighbor_gts_onehot[:self.knn_neighbors_num])
        # 直接用distance计算权重
        weights = self.get_weights(distances)
        # # change weights
        # neighbors_preds = torch.argmax(neighbors_probs, dim=-1)
        # pred_x = torch.argmax(probs_x)
        # # 邻居做错了，会乘以缩放因子0.5
        # factor1 = torch.where(neighbors_preds.cpu() == neighbor_gts.cpu(),
        #                       torch.tensor(1.0), torch.tensor(0.9))
        # # # 邻居和x预测不一样，会乘以缩放因子0.9
        # # factor2 = torch.where(neighbors_preds == pred_x,
        # #                       torch.tensor(1.0).to(self.device), torch.tensor(0.9).to(self.device))
        # weights = weights * factor1.unsqueeze(1).to(self.device)

        # # filter neighbors, only keep the neighbors whose pred is same as x
        # neighbors_preds = torch.argmax(neighbors_probs, dim=-1)
        # pred_x = torch.argmax(probs_x)
        # same_pred_ids = torch.nonzero(neighbors_preds == pred_x).squeeze(-1)
        # select_ids = same_pred_ids
        # select_ids = torch.arange(10,20)
        # select_ids = torch.arange(10)
        # pt_neighbors = (pt_neighbors[0][select_ids],
        #                 pt_neighbors[1][select_ids],
        #                 pt_neighbors[2][select_ids],
        #                 pt_neighbors[3][select_ids])
        # weights = weights[select_ids]
        return pt_neighbors, weights
    def pseudo_train_1(self,pt_neighbors, weights, pt_lambda):
        logits, probs, gts,gts_onehot = pt_neighbors
        delta_prob = pt_lambda * (weights*(gts_onehot-probs)).sum(dim=0)
        probs += delta_prob
        pt_neighbors = (logits, probs, gts, gts_onehot)
        return delta_prob, pt_neighbors
    def pseudo_train_2(self,pt_neighbors, weights, pt_lambda):
        logits, probs, gts,gts_onehot = pt_neighbors
        delta_logtis = pt_lambda * (weights*(gts_onehot-probs)).sum(dim=0)
        logits += delta_logtis
        probs = F.softmax(logits, dim=-1)
        pt_neighbors = (logits, probs, gts, gts_onehot)
        return delta_logtis, pt_neighbors
    def neighbor_decision_tree(self,inputs):
        # 断言，input_ids是1维的
        assert len(inputs.shape) == 1, "input_ids should be 1-dim"
        with torch.no_grad():
            valid_lenth = inputs.ne(self.tokenizer.pad_token_id).sum(-1)
            vecs, logits = self.get_encoded_vecs_and_logits(inputs)
            probs = F.softmax(logits, dim=-1)
            # 批量knn搜索
            vec_search = vecs[:valid_lenth - 1].cpu().detach().numpy()
            D, I = self.index.search(vec_search, self.knn_neighbors_num)
            example_dict = {
                'dis_vec': [],
                'dis_logits': [],
                'self_acc': [],
                'same_pred': [],
                'loss': [],
                'is_knn_useful': [],
                'is_pt_useful': [],
                'maybe_pt_useful': [],
            }
            for position in tqdm(range(valid_lenth - 1)):
                vec_x = vecs[position]
                logits_x = logits[position]
                probs_x = probs[position]
                pred_x = torch.argmax(probs_x)
                ground_truth_x = inputs[position + 1]
                # knn搜索邻居部分
                distances = D[position]
                neighbors_id = I[position]
                pt_neighbors, weights = self.init_pt_neighbors(vec_x, logits_x, probs_x, distances, neighbors_id)
                neighbor_logits, neighbor_probs, neighbor_gts, neighbor_gts_onehot = pt_neighbors
                # neighbor_logits = neighbor_logits.cpu().detach().numpy()
                # neighbor_probs = neighbor_probs.cpu().detach().numpy()
                # neighbor_gts = neighbor_gts.cpu().detach().numpy()
                for i in range(len(neighbor_logits)):
                    # vec,logits, gouund_truth, sf_logits
                    # vec_i,logits_i, ground_truth_i, sf_logits_i = pt_neighbors[i]
                    logits_i = neighbor_logits[i]
                    ground_truth_i = neighbor_gts[i]
                    probs_i = neighbor_probs[i]
                    pred_i = torch.argmax(probs_i)
                    is_knn_useful = ground_truth_i == ground_truth_x
                    # pt useful 有2种情况，1 是和knn一样，直接邻居的ground_truth和x一样，2是能降低x正确预测的同时，也降低了x的错误预测，并且提高了x第二项错误预测，但是要求第二项预测的概率不得高于正确预测的概率
                    is_pt_useful = is_knn_useful
                    maybe_pt_useful = is_knn_useful
                    if not is_knn_useful and pred_x != ground_truth_x:
                        prob_x_xgt = probs_x[ground_truth_x]
                        prob_x_xpred = probs_x[pred_x]
                        prob_x_igt = probs_x[ground_truth_i]
                        prob_i_xgt = probs_i[ground_truth_x]
                        prob_i_xpred = probs_i[pred_x]
                        if prob_i_xpred > prob_i_xgt:
                            maybe_pt_useful = True
                            if prob_x_igt < prob_x_xgt:
                               is_pt_useful = True
                    dis_vec = distances[i]
                    dis_logits = torch.dot(logits_x, logits_i)
                    self_acc = ground_truth_i == pred_i
                    same_pred = torch.argmax(probs_x) == torch.argmax(probs_i)
                    loss = F.cross_entropy(logits_i.unsqueeze(0), torch.tensor([ground_truth_i], dtype=torch.long).to(self.device))
                    example_dict['dis_vec'].append(dis_vec)
                    example_dict['dis_logits'].append(dis_logits.item())
                    example_dict['self_acc'].append(1 if self_acc else 0)
                    example_dict['same_pred'].append(1 if same_pred else 0)
                    example_dict['loss'].append(loss.item())
                    example_dict['is_knn_useful'].append(1 if is_knn_useful else 0)
                    example_dict['is_pt_useful'].append(1 if is_pt_useful else 0)
                    example_dict['maybe_pt_useful'].append(1 if maybe_pt_useful else 0)
        return example_dict
    
    # add myself
    def output_time(self):
        return f'ori_generation: {self.time_ori_generation}\npt_generation: {self.time_pt_generation}\npt_retrieval: {self.time_pt_retrieval}\nknm_generation: {self.time_knm_generation}\nknm_retrieval: {self.time_knm_retrieval}'

class CodeCompletionModel_Gpt2(CodeCompletionModel):
    pass

# add myself
class CodeCompletionModel_Llama(CodeCompletionModel):
    def __init__(self, model_class, model_name, config, tokenizer, vecs_path, knn_infos_path,
                 index_save_path, xgboost_path):
        super(CodeCompletionModel_Llama, self).__init__(model_class, model_name, config, tokenizer, vecs_path, knn_infos_path,
                    index_save_path, xgboost_path)
        
        self.lm_head = self.encoder.lm_head
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # if lm_head_path is not None:
        #     self.lm_head.load_state_dict(torch.load(lm_head_path))
        # else:
        #     self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.bias = torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
    def set_device(self, device):
        self.device = device
        self.lm_head.to(device)
        self.encoder.to(device)
        self.bias = self.bias.to(device)
    # def get_logits_from_vecs(self, vecs):
    #     with torch.no_grad():
    #         logits = self.lm_head(vecs)
    #     return logits
    def get_encoded_vecs_and_logits(self, inputs):
        # 关键函数，gpt2不用改，unixcoder需要改
        with torch.no_grad():
            # 如果是1维的，就扩展成2维的
            if len(inputs.shape) == 1:
                inputs_batch = inputs.unsqueeze(0)
                length = inputs_batch.size(-1)
                
                outputs = self.encoder(inputs_batch, output_hidden_states=True)
                
                last_hidden_states = outputs['hidden_states'][-1]
                logits = outputs['logits']
                # logits = self.lm_head(last_hidden_states)
                last_hidden_states = last_hidden_states.squeeze(0)
                logits = logits.squeeze(0)
            else:
                length = inputs.size(-1)
                outputs = self.encoder(inputs, output_hidden_states=True)
                
                last_hidden_states = outputs['hidden_states'][-1]
                # logits = self.lm_head(last_hidden_states)
                logits = outputs['logits']
        return last_hidden_states, logits

    def get_past_key_values(self, inputs, past_key_values=None):
        with torch.no_grad():
            if past_key_values is None:
                outputs = self.encoder(inputs, output_hidden_states=True)
            else:
                outputs = self.encoder(inputs,past_key_values=past_key_values,output_hidden_states=True)
            past_key_values = outputs.past_key_values
            last_hidden_states = outputs.hidden_states[-1]
            # logits = self.lm_head(last_hidden_states)
            logits = outputs.logits
        return logits, last_hidden_states, past_key_values


# add myself
class CodeCompletionModel_StarCoder(CodeCompletionModel):
    def __init__(self, model_class, model_name, config, tokenizer, vecs_path, knn_infos_path,
                 index_save_path, xgboost_path):
        super(CodeCompletionModel_StarCoder, self).__init__(model_class, model_name, config, tokenizer, vecs_path, knn_infos_path,
                    index_save_path, xgboost_path)
        
        self.lm_head = self.encoder.lm_head
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # if lm_head_path is not None:
        #     self.lm_head.load_state_dict(torch.load(lm_head_path))
        # else:
        #     self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.bias = torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
    def set_device(self, device):
        self.device = device
        self.lm_head.to(device)
        self.encoder.to(device)
        self.bias = self.bias.to(device)
    # def get_logits_from_vecs(self, vecs):
    #     with torch.no_grad():
    #         logits = self.lm_head(vecs)
    #     return logits

    # 似乎predict_line没有用到该函数
    def get_encoded_vecs_and_logits(self, inputs):
        # 关键函数，gpt2不用改，unixcoder需要改
        with torch.no_grad():
            # 如果是1维的，就扩展成2维的
            if len(inputs.shape) == 1:
                inputs_batch = inputs.unsqueeze(0)
                length = inputs_batch.size(-1)
                
                outputs = self.encoder(inputs_batch, output_hidden_states=True)
                
                last_hidden_states = outputs['hidden_states'][-1]
                logits = outputs['logits']
                # logits = self.lm_head(last_hidden_states)
                last_hidden_states = last_hidden_states.squeeze(0)
                logits = logits.squeeze(0)
            else:
                length = inputs.size(-1)
                outputs = self.encoder(inputs, output_hidden_states=True)
                
                last_hidden_states = outputs['hidden_states'][-1]
                # logits = self.lm_head(last_hidden_states)
                logits = outputs['logits']
        return last_hidden_states, logits

    def get_past_key_values(self, inputs, past_key_values=None):
        with torch.no_grad():
            if past_key_values is None:
                outputs = self.encoder(inputs, output_hidden_states=True)
            else:
                outputs = self.encoder(inputs,past_key_values=past_key_values,output_hidden_states=True)
            
            past_key_values = outputs.past_key_values
            last_hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
        return logits, last_hidden_states, past_key_values
    


class Beam1(object):
    def __init__(self, size, eos, device):
        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(0).to(device)]
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.nextYs[-1].view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode="floor")
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


class Beam(object):
    def __init__(self, size, sos, eos, device):
        self.size = size
        self.device = device
        # The score for each translation on the beam.
        # self.scores = self.tt.FloatTensor(size).zero_()
        self.scores = torch.zeros(size, dtype=torch.float32, device=self.device)

        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        # self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs = [torch.zeros(size, dtype=torch.int64, device=self.device)]

        self.nextYs[0][:] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        # batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        batch = torch.tensor(self.nextYs[-1], dtype=torch.int64, device=self.device).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] in self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] in self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] in self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] not in self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                tokens.append(tok)
                if tok in self._eos:
                    break
            sentence.append(tokens)
        return sentence