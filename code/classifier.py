import os
import json
import glob
import random

from utils import FeatureComputer, split_py_code, split_java_code

from numpy import vstack
import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_normal_
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tqdm import tqdm

from imblearn.over_sampling import SMOTE
import numpy as np

class ClassifierDataCollector(object):
    def __init__(self, l_model, l_model_type, s_model, s_model_type, tokenizer, device):
        self.l_model = l_model
        self.l_model_type = l_model_type

        self.s_model = s_model
        self.s_model_type = s_model_type

        self.tokenizer = tokenizer
        self.device = device

        self.l_model_last_layer = self.l_model.lm_head
        self.l_model_last_layer.register_forward_pre_hook(self.hook_fn_get_l_model_input)
        self.l_model_ll_embedding = None

        self.s_model_last_layer = self.s_model.lm_head
        self.s_model_last_layer.register_forward_pre_hook(self.hook_fn_get_s_model_input)
        self.s_model_ll_embedding = None

    def hook_fn_get_l_model_input(self, module, input):
        self.l_model_ll_embedding = input[0]

    def hook_fn_get_s_model_input(self, module, input):
        self.s_model_ll_embedding = input[0]
        
    def collect_parallel(self, dataset, output_dir, corpus, lang):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        embeddings = []
        features = []

        with torch.no_grad():
            for case in tqdm(dataset):
                case = json.loads(case)
                input = case['input'].replace(" <EOL> ", "\n")  # for py
                input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(self.device)

                if lang == 'python':
                    statements = split_py_code(input)
                elif lang == 'java':
                    statements = split_java_code(input)
                    
                featureComputer = FeatureComputer(
                    lang=lang,
                    context_statements=statements,
                    context_ids=input_ids,
                    corpus=corpus,
                    k=10
                )
                    
                gt = case['gt']
                if lang == 'python':
                    gt = gt + '\n'
                elif lang == 'java':
                    gt = ' ' + gt

                if self.l_model_type == 'deepseek-coder':
                    gt_ids = self.tokenizer(gt, return_tensors="pt").input_ids[:, 1:].to(self.device)   # [1:] get rid of bos_token
                # elif self.l_model_type == 'starcoder2':
                #     gt_ids = self.tokenizer(gt, return_tensors="pt").input_ids.to(self.device)

                # generate in parallel
                current_input_ids = torch.cat((input_ids, gt_ids[:, :-1]), dim=-1)
                s_model_output = self.generate_parallel(
                    model=self.s_model,
                    model_type=self.s_model_type,
                    input_ids=current_input_ids
                )
                l_model_output = self.generate_parallel(
                    model=self.l_model,
                    model_type=self.l_model_type,
                    input_ids=current_input_ids
                )

                gt_len = gt_ids.shape[1]
                for i in range(gt_len-1):
                    pos = -(i+1)
                    gt_token_id = gt_ids[:, pos]
                    s_token_id = s_model_output['generated_token_ids'][:, pos]
                    l_token_id = l_model_output['generated_token_ids'][:, pos]

                    if (s_token_id == gt_token_id and l_token_id != gt_token_id) or (s_token_id != gt_token_id and l_token_id == gt_token_id):
                        # get embedding
                        l_embedding = self.l_model_ll_embedding[:, pos].flatten(0, 1)
                        s_embedding = self.s_model_ll_embedding[:, pos].flatten(0, 1)
                        embedding = torch.cat((l_embedding, s_embedding), dim=-1)
                        embeddings.append(embedding)

                        # get feature
                        label = 1 if s_token_id == gt_token_id else 0
                        
                        # Prob
                        l_token_probs = l_model_output['all_token_probs'][:, pos]
                        s_token_probs = s_model_output['all_token_probs'][:, pos]
                        l_token_prob = l_model_output['generated_token_probs'][:, pos]
                        s_token_prob = s_model_output['generated_token_probs'][:, pos]
                        l_prob_of_s_token = l_token_probs[0][s_token_id.item()]
                        s_prob_of_l_token = s_token_probs[0][l_token_id.item()]
                        
                        # Type
                        l_word = self.tokenizer.decode(l_token_id)
                        s_word = self.tokenizer.decode(s_token_id)
                        l_word_type = featureComputer.get_word_type(l_word)
                        s_word_type = featureComputer.get_word_type(s_word)

                        # Frequency(word)
                        l_token_id_cnt = featureComputer.get_token_id_cnt(l_token_id)
                        s_token_id_cnt = featureComputer.get_token_id_cnt(s_token_id)
                        l_word = l_word.strip()
                        s_word = s_word.strip()
                        l_word_cnt = featureComputer.get_word_cnt(l_word)
                        s_word_cnt = featureComputer.get_word_cnt(s_word)
                        l_subword_cnt = featureComputer.get_subword_cnt(l_word)
                        s_subword_cnt = featureComputer.get_subword_cnt(s_word)
                        l_word_gauss_cnt = featureComputer.get_word_gauss_cnt(l_word)
                        s_word_gauss_cnt = featureComputer.get_word_gauss_cnt(s_word)
                        l_subword_gauss_cnt = featureComputer.get_subword_gauss_cnt(l_word)
                        s_subword_gauss_cnt = featureComputer.get_subword_gauss_cnt(s_word)

                        # Frequency(line)
                        gt_prefix = gt_ids[0, :pos]
                        l_statement_ids = torch.cat((gt_prefix, l_token_id), dim=-1)
                        s_statement_ids = torch.cat((gt_prefix, s_token_id), dim=-1)
                        l_statement_ids_cnt = featureComputer.get_statement_ids_cnt(l_statement_ids)
                        s_statement_ids_cnt = featureComputer.get_statement_ids_cnt(s_statement_ids)

                        # Frequency(line)
                        l_statement = self.tokenizer.decode(l_statement_ids).strip()
                        s_statement = self.tokenizer.decode(s_statement_ids).strip()
                        l_statement_cnt = featureComputer.get_statement_cnt(l_statement)
                        s_statement_cnt = featureComputer.get_statement_cnt(s_statement)

                        # Similarity
                        l_words = l_statement.split()
                        s_words = s_statement.split()
                        l_bm25_of_context = featureComputer.get_bm25_of_context(l_words)
                        s_bm25_of_context = featureComputer.get_bm25_of_context(s_words)
                        l_bm25_of_lastline = featureComputer.get_bm25_of_lastline(l_words)
                        s_bm25_of_lastline = featureComputer.get_bm25_of_lastline(s_words)
                        l_bm25_of_context_with_lastline = featureComputer.get_bm25_of_context_with_lastline(l_words)
                        s_bm25_of_context_with_lastline = featureComputer.get_bm25_of_context_with_lastline(s_words)

                        lastline = statements[-1]
                        # Frequency
                        l_multiline_cnt = featureComputer.get_multiline_cnt(lastline, l_statement)
                        s_multiline_cnt = featureComputer.get_multiline_cnt(lastline, s_statement)

                        # Retrieval
                        l_statement_cnt_in_corpus, s_statement_cnt_in_corpus, l_statement_cnt_in_corpus_with_lastline, s_statement_cnt_in_corpus_with_lastline = featureComputer.get_cnt_in_corpus(
                            lastline=lastline,
                            l_statement=l_statement,
                            s_statement=s_statement
                        )
                        l_avg_top_k_jaccard_with_lastline_in_corpus, s_avg_top_k_jaccard_with_lastline_in_corpus, l_top_k_jaccard_in_corpus, s_top_k_jaccard_in_corpus = featureComputer.get_jaccard_in_corpus(
                            lastline=lastline,
                            l_words=l_words,
                            s_words=s_words
                        )

                        # Length
                        len_input_ids = featureComputer.get_len_input_ids()
                        len_input_words = featureComputer.get_len_input_words()
                        len_generate_token_ids = gt_len - i
                        len_generate_words = min(len(l_words), len(s_words))
                        len_statements = featureComputer.get_len_statements()
                        len_lastline = featureComputer.get_len_lastline()

                        feature = {
                            'label': label,
                            # Length
                            'len_generate_token_ids': len_generate_token_ids,
                            'len_generate_words': len_generate_words,
                            'len_input_ids': len_input_ids,
                            'len_input_words': len_input_words,
                            'len_statements': len_statements,
                            'len_lastline': len_lastline,

                            # Prob
                            'l_token_prob': l_token_prob.item(),
                            's_token_prob': s_token_prob.item(),
                            'l_prob_of_s_token': l_prob_of_s_token.item(),
                            's_prob_of_l_token': s_prob_of_l_token.item(),

                            # Type
                            'l_word_type': l_word_type,
                            's_word_type': s_word_type,

                            # Token_id
                            'l_token_id': l_token_id.item(),
                            's_token_id': s_token_id.item(),

                            # Frequency
                            'l_token_id_cnt': l_token_id_cnt.item(),
                            's_token_id_cnt': s_token_id_cnt.item(),
                            'l_word_cnt': l_word_cnt,
                            's_word_cnt': s_word_cnt,
                            'l_subword_cnt': l_subword_cnt,
                            's_subword_cnt': s_subword_cnt,
                            'l_word_gauss_cnt': l_word_gauss_cnt,
                            's_word_gauss_cnt': s_word_gauss_cnt,
                            'l_subword_gauss_cnt': l_subword_gauss_cnt,
                            's_subword_gauss_cnt': s_subword_gauss_cnt,
                            'l_statement_ids_cnt': l_statement_ids_cnt,
                            's_statement_ids_cnt': s_statement_ids_cnt,
                            'l_statement_cnt': l_statement_cnt,
                            's_statement_cnt': s_statement_cnt,
                            'l_multiline_cnt': l_multiline_cnt,
                            's_multiline_cnt': s_multiline_cnt,

                            # Similarity
                            'l_bm25_of_context': l_bm25_of_context,
                            's_bm25_of_context': s_bm25_of_context,
                            'l_bm25_of_lastline': l_bm25_of_lastline,
                            's_bm25_of_lastline': s_bm25_of_lastline,
                            'l_bm25_of_context_with_lastline': l_bm25_of_context_with_lastline,
                            's_bm25_of_context_with_lastline': s_bm25_of_context_with_lastline,

                            # Retrieval
                            'l_statement_cnt_in_corpus': l_statement_cnt_in_corpus,
                            's_statement_cnt_in_corpus': s_statement_cnt_in_corpus,
                            'l_statement_cnt_in_corpus_with_lastline': l_statement_cnt_in_corpus_with_lastline,
                            's_statement_cnt_in_corpus_with_lastline': s_statement_cnt_in_corpus_with_lastline,

                            'l_avg_top_k_jaccard_with_lastline_in_corpus': l_avg_top_k_jaccard_with_lastline_in_corpus,
                            's_avg_top_k_jaccard_with_lastline_in_corpus': s_avg_top_k_jaccard_with_lastline_in_corpus,
                            'l_top_k_jaccard_in_corpus': l_top_k_jaccard_in_corpus,
                            's_top_k_jaccard_in_corpus': s_top_k_jaccard_in_corpus,
                        }
                        features.append(feature)
                            
            torch.save(embeddings, os.path.join(output_dir, f'embeddings.pt'))
            with open(os.path.join(output_dir, 'features.json'), 'w') as wf:
                for feature in features:
                    wf.write(json.dumps(feature))
                    wf.write('\n')


    def generate_parallel(self, model, model_type, input_ids):
        if model_type in ['deepseek-coder', 'starcoder2']:
            input_ids = input_ids.to(model.device)
            output = model(input_ids=input_ids,
                        return_dict=True,
                        use_cache=True
                        )

            logits = output.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output_probs, output_ids = torch.max(probs, dim=-1)
        else:
            raise NotImplementedError('Unknown model type')
        
        return {
            'generated_token_ids': output_ids,
            'all_token_probs': probs,
            'generated_token_probs': output_probs
        }    
            

class ClassifierMLP(nn.Module):
    def __init__(self, input_size, scale_size, model_type, dropout=0.2):
        super().__init__()

        self.feature_fusion = FeatureFusion(scale_size, model_type)

        self.fc1 = nn.Linear(input_size, 20)
        kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        self.bn1 = nn.BatchNorm1d(20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(20, 16)
        kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(16, 1)
        xavier_normal_(self.fc3.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding, feature):
        x = self.feature_fusion(embedding, feature)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    
class FeatureFusion(nn.Module):
    def __init__(self, scale_size, model_type):
        super().__init__()
        self.model_type = model_type
        if model_type == 'deepseek-coder':
            self.l_embedding_size = 4096
            self.s_embedding_size = 2048
        # elif model_type == 'starcoder2':
        #     self.l_embedding_size = 4608
        #     self.s_embedding_size = 3072

        self.scale_size = scale_size
        if self.scale_size > 0:
            self.fc_l = nn.Linear(self.l_embedding_size, scale_size)
            kaiming_uniform_(self.fc_l.weight)
            self.fc_s = nn.Linear(self.s_embedding_size, scale_size)
            kaiming_uniform_(self.fc_s.weight)

    def forward(self, embedding, feature):
        feature = self.feature_process(embedding[0].device, feature)

        # 统计信息
        len_generate_token_ids = feature['len_generate_token_ids']
        len_generate_words = feature['len_generate_words']
        len_input_ids = feature['len_input_ids']
        len_input_words = feature['len_input_words']
        len_statements = feature['len_statements']
        len_lastline = feature['len_lastline']

        # Prob
        l_token_prob = feature['l_token_prob']
        s_token_prob = feature['s_token_prob']
        l_prob_of_s_token = feature['l_prob_of_s_token']
        s_prob_of_l_token = feature['s_prob_of_l_token']

        # Type
        l_word_type = feature['l_word_type']
        s_word_type = feature['s_word_type']

        # Frequency
        l_token_id_cnt = feature['l_token_id_cnt'] 
        s_token_id_cnt = feature['s_token_id_cnt'] 
        l_word_cnt = feature['l_word_cnt'] 
        s_word_cnt = feature['s_word_cnt'] 
        l_subword_cnt = feature['l_subword_cnt'] 
        s_subword_cnt = feature['s_subword_cnt'] 
        l_word_gauss_cnt = feature['l_word_gauss_cnt']
        s_word_gauss_cnt = feature['s_word_gauss_cnt']
        l_subword_gauss_cnt = feature['l_subword_gauss_cnt']
        s_subword_gauss_cnt = feature['s_subword_gauss_cnt']
        l_statement_ids_cnt = feature['l_statement_ids_cnt'] 
        s_statement_ids_cnt = feature['s_statement_ids_cnt'] 
        l_statement_cnt = feature['l_statement_cnt'] 
        s_statement_cnt = feature['s_statement_cnt'] 
        l_multiline_cnt = feature['l_multiline_cnt'] 
        s_multiline_cnt = feature['s_multiline_cnt'] 

        # Similarity
        l_bm25_of_context = feature['l_bm25_of_context'] 
        s_bm25_of_context = feature['s_bm25_of_context'] 
        l_bm25_of_lastline = feature['l_bm25_of_lastline'] 
        s_bm25_of_lastline = feature['s_bm25_of_lastline'] 
        l_bm25_of_context_with_lastline = feature['l_bm25_of_context_with_lastline'] 
        s_bm25_of_context_with_lastline = feature['s_bm25_of_context_with_lastline'] 

        # Retrieval
        l_statement_cnt_in_corpus = feature['l_statement_cnt_in_corpus'] 
        s_statement_cnt_in_corpus = feature['s_statement_cnt_in_corpus'] 
        l_statement_cnt_in_corpus_with_lastline = feature['l_statement_cnt_in_corpus_with_lastline'] 
        s_statement_cnt_in_corpus_with_lastline = feature['s_statement_cnt_in_corpus_with_lastline'] 
        l_avg_top_k_jaccard_with_lastline_in_corpus = feature['l_avg_top_k_jaccard_with_lastline_in_corpus'] 
        s_avg_top_k_jaccard_with_lastline_in_corpus = feature['s_avg_top_k_jaccard_with_lastline_in_corpus'] 
        l_top_k_jaccard_in_corpus = feature['l_top_k_jaccard_in_corpus'] 
        s_top_k_jaccard_in_corpus = feature['s_top_k_jaccard_in_corpus'] 

        sum = torch.exp(l_token_prob) + torch.exp(s_token_prob) + torch.exp(l_prob_of_s_token) + torch.exp(s_prob_of_l_token)
        l_weight = (torch.exp(l_token_prob) + torch.exp(s_prob_of_l_token)) / sum
        s_weight = (torch.exp(s_token_prob) + torch.exp(l_prob_of_s_token)) / sum
        l_model_embedding = embedding[:, :self.l_embedding_size] * l_weight
        s_model_embedding = embedding[:, self.l_embedding_size:] * s_weight
        if self.scale_size > 0:
            l_model_embedding = self.fc_l(l_model_embedding)
            s_model_embedding = self.fc_s(s_model_embedding)

        fused_features = torch.cat(
            (
                # Embedding
                l_model_embedding,
                s_model_embedding,

                # Prob
                l_token_prob,
                s_token_prob,
                l_prob_of_s_token,
                s_prob_of_l_token,

                # Type
                l_word_type,
                s_word_type,

                # Length
                len_generate_token_ids,
                len_generate_words,
                len_lastline,

                # Frequency
                (l_token_id_cnt - s_token_id_cnt) / len_input_ids,
                (l_word_cnt - s_word_cnt) / len_input_words,
                (l_subword_cnt - s_subword_cnt) / len_input_words,

                (l_word_gauss_cnt - s_word_gauss_cnt),
                (l_subword_gauss_cnt - s_subword_gauss_cnt),

                # (l_statement_ids_cnt - s_statement_ids_cnt),
                l_statement_ids_cnt,
                s_statement_ids_cnt,
                (l_statement_cnt - s_statement_cnt) / len_statements,
                (l_multiline_cnt - s_multiline_cnt),

                # Similarity
                l_bm25_of_context - s_bm25_of_context,
                l_bm25_of_lastline - s_bm25_of_lastline,
                l_bm25_of_context_with_lastline - s_bm25_of_context_with_lastline,

                # Retrieval
                torch.sign(l_statement_cnt_in_corpus - s_statement_cnt_in_corpus),
                torch.sign(l_statement_cnt_in_corpus_with_lastline - s_statement_cnt_in_corpus_with_lastline),
                torch.sign(l_avg_top_k_jaccard_with_lastline_in_corpus - s_avg_top_k_jaccard_with_lastline_in_corpus),
                torch.sign(l_top_k_jaccard_in_corpus - s_top_k_jaccard_in_corpus)
            ), dim=-1
        )

        return fused_features

    def feature_process(self, device, feature):
        for key in feature.keys():
            val = feature[key].float()
            val = torch.unsqueeze(val, 1)
            val = val.to(device)
            feature[key] = val
            
        return feature

class ClassifierDataset(Dataset):
    def __init__(self, data_dir, device, smote_flag=False):
        embeddings_path = os.path.join(data_dir, 'embeddings.pt')
        self.x1 = torch.load(embeddings_path, map_location=device, weights_only=True)
        
        with open(os.path.join(data_dir, 'features.json'), 'r') as f:
            features = f.readlines()
            features = [json.loads(item) for item in features]
        
        labels = [item['label'] for item in features]
        self.y = [int(l) for l in labels]
        assert len(self.x1) == len(self.y), "Lengths of self.x1 and self.y do not match."

        self.x2 = features
        assert len(self.x2) == len(self.y), "Lengths of self.x2 and self.y do not match."

        self.X = []
        for e, f in zip(self.x1, self.x2):
            for key in f.keys():
                if key == 'label':
                    continue
                val = torch.tensor(f[key]).to(e.device).float()
                val = val.unsqueeze(0)
                e = torch.cat((e, val), dim=-1)
            
            self.X.append(e)

        if smote_flag:
            X = np.array([item.cpu().numpy() for item in self.X])
            y = np.array(self.y)
            smote = SMOTE(sampling_strategy=1.0)

            X_resampled, y_resampled = smote.fit_resample(X, y)
            X_resampled = [torch.tensor(item, device=self.X[0].device) for item in X_resampled]
            y_resampled = torch.tensor(y_resampled, device=self.X[0].device)
            self.X = X_resampled
            self.y = y_resampled

            x1_resampled = []
            x2_resampled = []
            for e_resampled in X_resampled:
                e_x1_resampled = e_resampled[:len(self.x1[0])]
                e_x2_resampled = {}

                for key, val in zip(self.x2[0].keys(), e_resampled[len(self.x1[0])-1:]):
                    if key == 'label':
                        continue
                    e_x2_resampled[key] = val.item()

                x1_resampled.append(e_x1_resampled)
                x2_resampled.append(e_x2_resampled)
            
            self.x1 = x1_resampled
            self.x2 = x2_resampled


    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        return [self.x1[idx], self.x2[idx], self.y[idx]]
    
class Classifier():
    def __init__(self, input_size, scale_size, dropout, train_dir, validation_dir, model_path, model_type, device, logger):
        self.model = ClassifierMLP(input_size, scale_size, model_type, dropout)

        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.logger = logger

        if model_path is not None: # do_eval
            self.model_path = model_path
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            logger.info(f'Loading classifier model from {model_path}')
        
        self.device = device
        self.model.to(self.device)

    def train_model(self, output_dir, epoch, save_step, batch_size, lr, weight_decay=1e-5, lr_step=15, lr_gamma=0.5, threshold=0.5):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        train_dataset = ClassifierDataset(
            data_dir=self.train_dir,
            device=self.device,
            smote_flag=True
            # smote_flag=False
        )
        self.logger.info(f'Loading train data from {self.train_dir}')
        self.eval_dataset(threshold=threshold, dataset=train_dataset)

        validation_dataset = ClassifierDataset(
            data_dir=self.validation_dir,
            device=self.device,
            smote_flag=False
        )
        self.logger.info(f'Loading validation data from {self.validation_dir}')

        self.logger.info('Start training ...')        
        for ep in tqdm(range(epoch)):
            total_loss = 0
            total_batches = 0
    
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            for i, (embeddings, features, targets) in enumerate(train_dataloader):
                total_batches += 1
                embeddings = embeddings.to(self.device)
                targets = targets.float().to(self.device)

                optimizer.zero_grad()
                yhat = self.model(embeddings, features).view(-1)
                loss = criterion(yhat, targets)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            scheduler.step()
            
            if ep > 0 and ep % save_step == 0:
                avg_loss = total_loss / total_batches
                self.logger.info(f'epoch: {ep}, avg-loss: {avg_loss}')
                torch.save(self.model.state_dict(), os.path.join(output_dir, f'ckpt{ep}.pth'))
                self.logger.info(f'save ckpt{ep}.pth in {output_dir}')
                
                train_acc, train_loss, train_precision, train_recall = self.eval_dataset(threshold, train_dataset)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                self.logger.info(f'For training dataset, avg-acc = {train_acc}, avg-precision = {train_precision}, avg-recall = {train_recall}, avg-loss = {train_loss}')

                val_acc, val_loss, val_precision, val_recall = self.eval_dataset(threshold, validation_dataset)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                self.logger.info(f'For validation dataset, avg-acc = {val_acc}, avg-precision = {val_precision}, avg-recall = {val_recall}, avg-loss = {val_loss}')

        with open(os.path.join(output_dir, 'stat.txt'), 'w') as wf:
            wf.write(f'train_losses = {train_losses}\n')
            wf.write(f'train_accs = {train_accs}\n')
            wf.write(f'val_losses = {val_losses}\n')
            wf.write(f'val_accs = {val_accs}\n')

    def eval_model(self, threshold, data_dir):
        evaluation_dataset = ClassifierDataset(
            data_dir=data_dir,
            device=self.device,
            smote_flag=False
        )

        avg_acc, avg_loss, avg_precision, avg_recall = self.eval_dataset(threshold, evaluation_dataset)
        self.logger.info(f'For evaluation dataset, avg-acc = {avg_acc}, avg-precision = {avg_precision}, avg-recall = {avg_recall}, avg-loss = {avg_loss}')

    def eval_dataset(self, threshold, dataset):
        self.model.eval()
        accs, precisions, recalls, losses = [], [], [], []
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        acc, precision, recall, loss = self.eval_dataloader(dataloader, threshold)

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        losses.append(loss)

        self.model.train()

        if len(accs) != 0:
            return sum(accs) / len(accs), sum(losses) / len(losses), sum(precisions) / len(precisions), sum(recalls) / len(recalls)
        else:
            return 0, 0, 0, 0

    def eval_dataloader(self, dataloader, threshold):
        actual_predictions, predictions, actuals = [], [], []

        criterion = nn.BCELoss()
        total_loss = 0

        total_batches = 0
        for i, (embedding, feature, target) in enumerate(dataloader):
            embedding = embedding.to(self.device)
            target = target.float().to(self.device)
            yhat = self.model(embedding, feature).view(-1)
            
            total_batches += 1
            total_loss += criterion(yhat, target).item()
            
            yhat = yhat.float().detach().cpu().numpy()
            actual = target.float().cpu().numpy()

            actual_predictions.append(yhat)
            yhat = 1 if yhat >= threshold else 0
            
            predictions.append(yhat)
            actuals.append(actual)

        predictions, actuals = vstack(predictions), vstack(actuals)
                
        acc = accuracy_score(actuals, predictions)    
        precision = precision_score(actuals, predictions)
        recall = recall_score(actuals, predictions)
        return acc, precision, recall, total_loss / total_batches
    
    
    def predict(self, embedding, feature):
        self.model.eval()
        embedding = embedding.to(self.device)
        yhat = self.model(embedding, feature).view(-1)
        return yhat