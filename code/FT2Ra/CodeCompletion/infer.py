import logging
import os
import pickle

import torch

from tqdm import tqdm

from util import get_online_model, set_seed, get_dataloader_line, get_dataloader_token, \
    get_online_model_knm, get_knm_base_ratio, calc_eval_lines, calc_eval_tokens, calc_eval_lines_CodeXGlue, \
    get_dataloader_token_reacc, show_input, show_gt, show_pred


def knn_online_infer_line(dataname, modeltype, save_dir, datadir, is_finetuned):
    
    # add myself
    language = 'java'       # 'java' or 'python'
    knm_flag = True

    # for multi-step inference
    max_pt_step = 20
    pt_step = 6
    pt1_lambda = 1.5
    if dataname == "javaCorpus":
        if is_finetuned:
            pt2_lambda = 0.5
        else:
            pt2_lambda = 3
    elif dataname == "py150":
        if is_finetuned:
            pt2_lambda = 0.8
        else:
            pt2_lambda = 5
    else:
        if is_finetuned:
            pt2_lambda = 3
        else:
            pt2_lambda = 6
    # knn_lambda = 0.1
    knn_lambda = 0.2
    knn_neighbors_num = 20
    # print("pt1_lambda:{},pt2_lambda:{},knn_neighbors_num:{}".format(pt1_lambda, pt2_lambda, knn_neighbors_num))
    nprobe = 20  # 每个查询的聚类中心的数量,越大越准，但是越慢，默认是1

    beam_size = 5 # comment myself: 并不影响什么

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned", dataname)
    if modeltype == "gpt2":
        pretrained = "../../codegpt/CodeGPT-small-java-adaptedGPT2"
        if language == 'python':
            pretrained = "../../codegpt/CodeGPT-small-py-adaptedGPT2"
    
    # add myself
    elif modeltype == 'llama':
        pretrained = "../../model/deepseek/deepseek-coder-6.7b-base"
    elif modeltype == 'starcoder':
        pretrained = "../../starcoder2/starcoder2-7b"
                

    if not knm_flag:
        model, tokenizer, config = get_online_model(modeltype, pretrained, knn_save_dir)
    else:
        # for knm
        model, tokenizer, config = get_online_model_knm(modeltype, pretrained, knn_save_dir)
        knm_lambda_path = os.path.join(knn_save_dir, "wrong_ratio.pkl")
        with open(knm_lambda_path, 'rb') as f:
            knm_lambda = pickle.load(f)
            print("knm_lambda:{}".format(knm_lambda))
            knn_lambda = knm_lambda
            model.set_knm_lambda(knm_lambda)


    datatype = "test"
    # datatype = "testreacc"
    # datatype = "testbm25"
    batch_size = 1
    test_dataloader, test_dataset = get_dataloader_line(datadir, modeltype, dataname, datatype, tokenizer, batch_size,
                                                         save_dir)
    device = torch.device("cuda:0")
    model.set_device(device)
    model.set_pt_config(max_pt_step, knn_neighbors_num, nprobe, knn_lambda, pt1_lambda, pt2_lambda)
    model.set_line_config(beam_size, pt_step)
    model.to(device)
    model.eval()
    set_seed(42)

    ori_EM = 0.0
    ori_edit_sim = 0.0
    knn_EM = 0.0
    knn_edit_sim = 0.0
    pt_EM = 0.0
    pt_edit_sim = 0.0
    total = 0

    # update myself
    if language == 'python':
        if modeltype == 'llama':
            break_ids = [tokenizer.sep_token_id, 185]
        elif modeltype == 'starcoder':
            break_ids = [tokenizer.sep_token_id, 222]
    else:
        if modeltype == 'gpt2':
            break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
        elif modeltype == 'llama':
            break_ids = [6203, 507, 611]    # ; { }
        elif modeltype == 'starcoder':
            break_ids = [2098, 320, 339]

    ori_preds = []
    knn_preds = []
    pt_preds = []
    
    for batch in tqdm(test_dataloader):
    # for batch in test_dataloader:
        # 因为按token处理，所以batch为1
        assert batch[0].size(0) == 1
        (id, inputs, gt) = batch

        inputs = inputs.to(device)
        gt = gt[0]
        with torch.no_grad():
            if knm_flag:
                pred_line_dict = model.predict_line_new(inputs,break_ids,knn_need=True,pt_need=False)
            else:
                pred_line_dict = model.predict_line_new(inputs,break_ids,knn_need=False,pt_need=True)
            # pred_line_dict = model.predict_line_new(inputs,break_ids,knn_need=False,pt_need=False)
            flag = False

            # ori_pred_line = pred_line_dict["origin"]
            # EM, edit_sim, ori_pred_text = calc_eval_lines_CodeXGlue(ori_pred_line, gt, tokenizer, dataname, modeltype, language, flag)
            # ori_EM += EM
            # ori_edit_sim += edit_sim
            # ori_preds.append(ori_pred_text)

            knn_pred_line = pred_line_dict["knn"]
            EM, edit_sim, knn_pred_text = calc_eval_lines_CodeXGlue(knn_pred_line, gt, tokenizer, dataname, modeltype, language, flag)
            knn_EM += EM
            knn_edit_sim += edit_sim
            knn_preds.append(knn_pred_text)

            # pt_pred_lines = pred_line_dict["pt"]
            # EM, edit_sim, pt_pred_text = calc_eval_lines_CodeXGlue(pt_pred_lines, gt, tokenizer, dataname, modeltype, language, flag)
            # pt_EM += EM
            # pt_edit_sim += edit_sim
            # pt_preds.append(pt_pred_text)


        total += 1
    # total = len(test_dataloader)

    print("final-total:{}==========================".format(total))
    print("ori_EM:{}".format(ori_EM / total))
    print("ori_edit_sim:{}".format(ori_edit_sim / total))
    print("knn_EM:{}".format(knn_EM / total))
    print("knn_edit_sim:{}".format(knn_edit_sim / total))
    print("pt_EM:{}".format(pt_EM / total))
    print("pt_edit_sim:{}".format(pt_edit_sim / total))
    print("pt1_lambda:{},pt2_lambda:{},knn_neighbors_num:{}".format(pt1_lambda, pt2_lambda, knn_neighbors_num))

    print(model.output_time())

    output_dir = os.path.join(save_dir, f'preds-{dataname}-fim')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'FT2Ra-ori.txt'), 'w+') as wf:
        for pred in ori_preds:
            if '\n' in pred:
                pred = pred[:pred.find('\n')]
            wf.write(pred)
            wf.write('\n')

    if knm_flag:
        with open(os.path.join(output_dir, 'FT2Ra-knm.txt'), 'w+') as wf:
            for pred in knn_preds:
                if '\n' in pred:
                    pred = pred[:pred.find('\n')]
                wf.write(pred)
                wf.write('\n')
    else:
        with open(os.path.join(output_dir, 'FT2Ra-pt_knn.txt'), 'w+') as wf:
            for pred in pt_preds:
                if '\n' in pred:
                    pred = pred[:pred.find('\n')]
                wf.write(pred)
                wf.write('\n')



def knn_online_infer_token(dataname, save_dir, datadir, is_finetuned, modeltype):
    # modeltype = "gpt2"
    max_pt_step = 10
    pt_step = 4
    # for special inference
    pt1_lambda = 1.5
    pt2_lambda = 3
    pt2_lambdas = [6, 4.5, 3.5, 3.5, 2.4, 2, 1.8, 1, 0.75, 0.4, 0.3]
    pt2_lambdas = [6, 4.5, 3, 2, 1.2, 0.8, 0.6, 0.45, 0.25, 0.2, 0.2]
    if not is_finetuned:
        pt1_lambda = 3
        pt2_lambda = 6
    # for javaCorpus inference
    if dataname == "javaCorpus":
        pt1_lambda = 0.2
        pt2_lambda = 0.5
        if not is_finetuned:
            pt1_lambda = 1.5
            pt2_lambda = 3
    # for py150 inference
    elif dataname == "py150":
        pt1_lambda = 0.4
        pt2_lambda = 0.8
        if not is_finetuned:
            pt1_lambda = 2.5
            pt2_lambda = 5

    knn_lambda = 0.1
    knn_neighbors_num = 20

    # print("pt1_lambda:{},pt2_lambda:{},knn_neighbors_num:{}".format(pt1_lambda,pt2_lambda,knn_neighbors_num))
    nprobe = 20  # 每个查询的聚类中心的数量,越大越准，但是越慢，默认是1
    beam_size = 4

    knn_save_dir = os.path.join(save_dir, "knnCache", modeltype, "unfinetuned", dataname)
    if modeltype == "gpt2":
        if dataname == "py150":
            # pretrained = "microsoft/CodeGPT-small-py-adaptedGPT2"
            pretrained = "../../codegpt/CodeGPT-small-py-adaptedGPT2"
        else:
            # pretrained = "microsoft/CodeGPT-small-java-adaptedGPT2"
            pretrained = "../../codegpt/CodeGPT-small-java-adaptedGPT2"

    # add myself
    elif modeltype == 'llama':
        pretrained = "../../model/deepseek/deepseek-coder-6.7b-base"
                
    model, tokenizer, config = get_online_model(modeltype, pretrained, knn_save_dir)
    # for knm
    # model, tokenizer, config = get_online_model_knm(modeltype, pretrained, knn_save_dir)
    # knm_lambda_path = os.path.join(knn_save_dir, "wrong_ratio.pkl")
    # with open(knm_lambda_path, 'rb') as f:
    #     knm_lambda = pickle.load(f)
    #     print("knm_lambda:{}".format(knm_lambda))
    #     knn_lambda = knm_lambda
    #     model.set_knm_lambda(knm_lambda)

    datatype = "test"
    # datatype = "train"
    batch_size = 1
    test_dataloader, test_dataset = get_dataloader_token(datadir, modeltype, dataname, datatype, tokenizer, batch_size,
                                                        save_dir)
    device = torch.device("cuda:0")
    model.set_device(device)
    model.set_pt_config(max_pt_step, knn_neighbors_num, nprobe, knn_lambda, pt1_lambda, pt2_lambda)
    model.set_line_config(beam_size, pt_step)
    model.to(device)
    model.eval()
    set_seed(42)

    ori_total = 0
    ori_correct = 0.0
    knn_total = 0
    knn_correct = 0.0
    pt1_totals = []
    pt2_totals = []
    pt1_corrects = []
    pt2_corrects = []
    for step in range(max_pt_step):
        pt1_totals.append(0)
        pt2_totals.append(0)
        pt1_corrects.append(0.0)
        pt2_corrects.append(0.0)
    # model.set_neighbor_select_rule("deep-finetuned")
    for step, batch in tqdm(enumerate(test_dataloader)):
        # 因为按token处理，所以batch为1
        assert batch[0].size(0) == 1
        (id, inputs) = batch
        inputs = inputs.squeeze(0).to(device)
        with torch.no_grad():
            # pred_ids_dict = model.predict_tokens_knm(inputs)
            pred_ids_dict = model.predict_tokens(inputs)
            ori_pred_ids = pred_ids_dict["origin"]
            knn_pred_ids = pred_ids_dict["knn"]
            total,correct = calc_eval_tokens(ori_pred_ids, inputs, tokenizer)
            ori_total += total
            ori_correct += correct
            total, correct = calc_eval_tokens(knn_pred_ids, inputs, tokenizer)
            knn_total += total
            knn_correct += correct
            for pt_step in range(max_pt_step):
                # pt_pred_ids = pred_ids_dict["pt1_"+str(pt_step)]
                # total,correct = calc_eval_tokens(pt_pred_ids, inputs, tokenizer)
                # pt1_totals[pt_step] += total
                # pt1_corrects[pt_step] += correct
                pt_pred_ids = pred_ids_dict["pt2_"+str(pt_step)]
                total,correct = calc_eval_tokens(pt_pred_ids, inputs, tokenizer)
                pt2_totals[pt_step] += total
                pt2_corrects[pt_step] += correct
            # if step % 50 == 0 or step < 10:
            #     print("step:{},total:{}==========================".format(step,ori_total))
            #     print("ori_percent:{}".format( ori_correct / ori_total))
            #     print("knn_percent:{}".format(knn_correct / knn_total))
            #     # for pt_step in range(max_pt_step):
            #     #     print("pt_step:{}, percent:{}".format(pt_step, pt1_corrects[pt_step] / pt1_totals[pt_step]))
            #     for pt_step in range(max_pt_step):
            #         print("pt_step:{}, percent:{}".format(pt_step, pt2_corrects[pt_step] / pt2_totals[pt_step]))
    print("final-total:{}==========================".format(ori_total))
    print("ori_percent:{}".format(ori_correct / ori_total))
    print("knn_percent:{}".format(knn_correct / knn_total))
    for pt_step in range(max_pt_step):
        print("pt_step:{}, percent:{}".format(pt_step, pt2_corrects[pt_step] / pt2_totals[pt_step]))
    print("data_name:{},modeltype:{}".format(dataname,modeltype))
    print("pt1_lambda:{},pt2_lambda:{},knn_neighbors_num:{}".format(pt1_lambda, pt2_lambda, knn_neighbors_num))


def work_on_line(save_dir, datadir):
    data_names = [
        # 'spring'
        'android'
        # 'django'
        # 'flask'
        ]
    
    for dataname in data_names:
        print("====================================")
        print("data_name:{}".format(dataname))
        # knn_online_infer_line(dataname, "gpt2", save_dir, datadir, False)
        knn_online_infer_line(dataname, "llama", save_dir, datadir, False)
        # knn_online_infer_line(dataname, "starcoder", save_dir, datadir, False)

def work_on_token(save_dir, datadir):
    data_names = [
                
                  ]
    for dataname in data_names:
        print("====================================")
        print("data_name:{}".format(dataname))
        # knn_online_infer_token(dataname, save_dir, datadir, False, "gpt2")
        # knn_online_infer_token(dataname, save_dir, datadir, False, "llama")  # add myself



import argparse

def main(save_dir, datadir):
    # work_on_token(save_dir, datadir)
    work_on_line(save_dir, datadir)

    device = torch.device('cuda:0')
    input = 'def sum ( a , b ) :\n'
    
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('../../model/deepseek/deepseek-coder-6.7b-base', device_map=device)
    tokenizer = AutoTokenizer.from_pretrained('../../model/deepseek/deepseek-coder-6.7b-base')
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

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    main(args.save_dir, args.datadir)
