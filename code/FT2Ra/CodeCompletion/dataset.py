# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import os
import pickle
import gc
import json

import torch
from torch.utils.data import Dataset
import logging

from transformers import RobertaTokenizer, GPT2Tokenizer, LlamaTokenizerFast, CodeLlamaTokenizerFast, GPT2TokenizerFast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LineCompletionDataset(Dataset):
    def __init__(self, tokenizer, output_dir, datafile , block_size, data_tag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cached_file = os.path.join(output_dir, data_tag)
        if os.path.exists(cached_file):
        # if os.path.exists(cached_file) and False:
            with open(cached_file, 'rb') as handle:
                self.inputs,self.gt = pickle.load(handle)
        else:
            self.inputs = []
            self.gt = []
            with open(datafile) as f:
                data = f.readlines()
            length = len(data)
            logger.info("Data size: %d" % (length))
            token_cnt = 0
            for idx, x in enumerate(data):
                x = json.loads(x)
                inputs = x["input"]    
                # inputs = x["input"].replace(" <EOL> ", '\n')    # update myself
                gt = x["gt"]
                try:
                    token_inputs = tokenizer.encode(inputs)
                    token_gt = tokenizer.encode(gt)
                    token_cnt += len(token_gt)
                    if isinstance(tokenizer, RobertaTokenizer):
                        token_inputs = token_inputs[1:-1]
                    if len(token_inputs) > block_size:
                        token_inputs = token_inputs[-block_size:]
                    self.inputs.append(token_inputs)
                    self.gt.append(gt)
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"lines: {len(self.inputs)}")
            logger.info(f"tokens: {token_cnt}")
            # del input_ids
            # gc.collect()
            with open(cached_file, 'wb') as handle:
                pickle.dump((self.inputs,self.gt), handle, protocol=pickle.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long),torch.tensor(self.inputs[item]),self.gt[item]

class TokenCompletionDatasetSpecial(Dataset):
    def __init__(self, tokenizer, output_dir, datafile , block_size, data_tag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cached_file = os.path.join(output_dir, data_tag)
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
        else:
            self.inputs = []
            with open(datafile) as f:
                data = f.readlines()
            length = len(data)
            logger.info("Data size: %d" % (length))
            token_cnt = 0
            for idx, x in enumerate(data):
                x = json.loads(x)
                x = ' '.join(x)
                x = x.strip()

                # x = x.replace(' <EOL> ', '\n')      # add myself
                
                assert x.startswith("<s>") and x.endswith("</s>")
                try:
                    if isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast):        # add GPT2TokenizerFast myself
                        sample = tokenizer.encode(x)
                        if len(sample) > block_size:
                            sample = sample[:block_size-1]+[tokenizer.eos_token_id]
                        token_cnt += len(sample)
                        pad_len = block_size - len(sample)
                        sample += [tokenizer.pad_token_id] * pad_len
                        self.inputs.append(sample)
                    elif isinstance(tokenizer, RobertaTokenizer):
                        # source = x
                        # source_tokens = [x for x in tokenizer.tokenize(source) if x != '\u0120']
                        # source_tokens = ["<s>", "<decoder-only>", "</s>"] + source_tokens[-(block_size - 3):]
                        # source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                        # padding_length = block_size - len(source_ids)
                        # source_ids += [tokenizer.pad_token_id] * padding_length
                        sample = tokenizer.encode(x)
                        sample = sample[1:-1]
                        if len(sample) > block_size:
                            sample = sample[:block_size-1]+[tokenizer.eos_token_id]
                        token_cnt += len(sample)
                        pad_len = block_size - len(sample)
                        sample += [tokenizer.pad_token_id] * pad_len
                        self.inputs.append(sample)
                    
                    # Add myself
                    elif isinstance(tokenizer, LlamaTokenizerFast):
                        sample = tokenizer.encode(x)
                        if len(sample) > block_size:
                            sample = sample[:block_size-1]+[tokenizer.eos_token_id]
                        token_cnt += len(sample)
                        pad_len = block_size - len(sample)
                        sample += [tokenizer.pad_token_id] * pad_len
                        self.inputs.append(sample)
                    elif isinstance(tokenizer, CodeLlamaTokenizerFast):
                        sample = tokenizer.encode(x)
                        sample = sample[1:-1]
                        if len(sample) > block_size:
                            sample = sample[:block_size-1]+[tokenizer.eos_token_id]
                        token_cnt += len(sample)
                        pad_len = block_size - len(sample)
                        sample += [tokenizer.pad_token_id] * pad_len
                        self.inputs.append(sample)
                    

                    else:
                        raise NotImplementedError
                except Exception as e:
                    print(e)
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"lines: {len(self.inputs)}")
            logger.info(f"tokens: {token_cnt}")
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        if item is None or self.inputs[item] is None:
            print(item)
            print(self.inputs[item])
        return torch.tensor(item, dtype=torch.long),torch.tensor(self.inputs[item])
class TokenCompletionDatasetReaccTrain(Dataset):
    def __init__(self, tokenizer, output_dir, datafile, block_size, data_tag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cached_file = os.path.join(output_dir, data_tag)
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                (self.inputs,self.context_lens) = pickle.load(handle)
                logger.info(f"load from {cached_file}")
        else:
            self.inputs = []
            self.context_lens = []
            with open(datafile) as f:
                data = f.readlines()
            length = len(data)
            logger.info("Data size: %d" % (length))
            token_cnt = 0
            crop_cnt = 0
            for idx, x in enumerate(data):
                # if idx >1000:
                #     break
                x = json.loads(x)
                # inputs = x["input"]
                retrieval = x["retrieval"]
                # gt = x["gt"]
                try:
                    token_inputs = tokenizer.encode(retrieval)
                    context_len = 0
                    if len(token_inputs) > block_size:
                        token_inputs = token_inputs[-block_size:]
                        crop_cnt += 1
                    token_cnt += len(token_inputs)
                    pad_len = block_size - len(token_inputs)
                    token_inputs += [tokenizer.pad_token_id] * pad_len
                    self.inputs.append(token_inputs)
                    self.context_lens.append(context_len)
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"lines: {len(self.inputs)}, crop: {crop_cnt}")
            logger.info(f"tokens: {token_cnt}")
            # del input_ids
            # gc.collect()
            with open(cached_file, 'wb') as handle:
                pickle.dump((self.inputs,self.context_lens), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long), torch.tensor(self.inputs[item]), self.context_lens[item]
class TokenCompletionDatasetReacc(Dataset):
    def __init__(self, tokenizer, output_dir, datafile, block_size, data_tag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cached_file = os.path.join(output_dir, data_tag)
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                (self.inputs,self.context_lens) = pickle.load(handle)
                logger.info(f"load from {cached_file}")
        else:
            self.inputs = []
            self.context_lens = []
            with open(datafile) as f:
                data = f.readlines()
            length = len(data)
            logger.info("Data size: %d" % (length))
            token_cnt = 0
            crop_cnt = 0
            for idx, x in enumerate(data):
                if idx >1000:
                    break
                x = json.loads(x)
                inputs = x["input"]
                retrieval = x["retrieval"]
                gt = x["gt"]
                try:
                    token_inputs = tokenizer.encode(inputs)
                    token_gt = tokenizer.encode(gt)
                    token_gt = token_gt+ tokenizer.encode("</s>")
                    token_cnt += len(token_gt)
                    context_len = len(token_inputs)
                    token_inputs = token_inputs + token_gt
                    if len(token_inputs) > block_size:
                        token_inputs = token_inputs[-block_size:]
                        crop_cnt += 1
                        context_len = block_size-len(token_gt)
                    pad_len = block_size - len(token_inputs)
                    token_inputs += [tokenizer.pad_token_id] * pad_len
                    self.inputs.append(token_inputs)
                    self.context_lens.append(context_len)
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"lines: {len(self.inputs)}, crop: {crop_cnt}")
            logger.info(f"tokens: {token_cnt}")
            # del input_ids
            # gc.collect()
            with open(cached_file, 'wb') as handle:
                pickle.dump((self.inputs,self.context_lens), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long), torch.tensor(self.inputs[item]), self.context_lens[item]
class TokenCompletionDataset(Dataset):
    def __init__(self, tokenizer, output_dir, datafile, block_size, data_tag):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cached_file = os.path.join(output_dir, data_tag)
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)
                logger.info(f"load from {cached_file}")
        else:
            self.inputs = []
            with open(datafile) as f:
                data = f.readlines()
            length = len(data)
            logger.info("Data size: %d" % (length))
            token_cnt = 0
            crop_cnt = 0
            for idx, x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    sample = tokenizer.encode(x)
                    if len(sample) > block_size:
                        sample = [tokenizer.bos_token_id] + sample[-(block_size-1):]
                        crop_cnt += 1
                    token_cnt += len(sample)
                    pad_len = block_size - len(sample)
                    sample += [tokenizer.pad_token_id] * pad_len
                    self.inputs.append(sample)
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"lines: {len(self.inputs)}, crop: {crop_cnt}")
            logger.info(f"tokens: {token_cnt}")
            # del input_ids
            # gc.collect()
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(item, dtype=torch.long), torch.tensor(self.inputs[item])
