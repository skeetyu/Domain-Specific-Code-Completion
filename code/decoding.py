import time
import random

import torch
from torch import nn

from utils import FeatureComputer, split_py_code, split_java_code

class Generator:
    def __init__(self, l_model, l_model_type, tokenizer, device, lang, corpus, with_sd, s_model, s_model_type, with_classifier, classifier, collector):
        self.l_model = l_model
        self.l_model_type = l_model_type
        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang

        self.nextline = False

        if corpus is not None and len(corpus) > 0:
            self.corpus = corpus
            
        self.with_sd = with_sd
        if with_sd:
            self.s_model = s_model
            self.s_model_type = s_model_type

        self.with_classifier = with_classifier
        if with_classifier:
            self.classifier = classifier
            self.classifier.model.eval()
            self.collector = collector

        if self.l_model_type == 'deepseek-coder':
            self.vocabulary_size = 32256
        # elif self.l_model_type == 'starcoder2':
        #     self.vocabulary_size = 49152

    def generate(self, model, model_type, past_key_values, input_ids):
        if model_type in ['deepseek-coder', 'starcoder2']:
            input_ids = input_ids.to(self.device)
            output = model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True
                        )
            logits = output.logits
            past_key_values = output.past_key_values

            probs = torch.nn.functional.softmax(logits, dim=-1)

            output_probs, output_ids = torch.max(probs, dim=-1)
        else:
            raise NotImplementedError('Unknown model type')
        
        return {
            'ids': output_ids,
            'probs': output_probs,
            'all_token_probs': probs,
            'past_key_values': past_key_values
        }


    def baseline_decoding(self, input_ids, max_gen_len):
        generate_ids = torch.empty([input_ids.size(0), max_gen_len], dtype=torch.long, device=self.device)

        current_input_ids = input_ids.to(self.device)
        past_key_values = None

        with torch.no_grad():
            for it in range(max_gen_len):
                output = self.generate(
                    model=self.l_model,
                    model_type=self.l_model_type,
                    past_key_values=past_key_values,
                    input_ids=current_input_ids
                )
                
                generate_ids[:, it] = output['ids'][:, -1]
                past_key_values = output['past_key_values']

                current_input_ids = output['ids'][:, -1:]

                if current_input_ids.item() == self.tokenizer.pad_token_id or current_input_ids.item() == self.tokenizer.eos_token_id:
                    break
                elif self.lang == 'python' and self.tokenizer.decode(current_input_ids.item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(current_input_ids.item()).strip() in ['{', ';', '}']:
                    break

        gen_len = it+1
        generate_ids = generate_ids[:, :gen_len]
        pred = self.tokenizer.decode(generate_ids[0]).strip()
        if '\n' in pred:
            pred = pred[:pred.find('\n')]
        if '<｜end▁of▁sentence｜' in pred:
            pred = pred[:pred.find('<｜end▁of▁sentence｜')]

        return pred.strip()
    
    def speculative_decoding(self, input_ids, max_gen_len, max_draft_len, th_classifier_prob):
        generate_ids = torch.empty([input_ids.size(0), max_gen_len], dtype=torch.long, device=self.device)

        context = self.tokenizer.decode(input_ids[0])
        context = context.replace('<｜begin▁of▁sentence｜>', '')
        if self.lang == 'python':
            context_statements = split_py_code(context)
        elif self.lang == 'java':
            context_statements = split_java_code(context)
        
        if not self.nextline:
            self.gt_prefix = context_statements[-1]
            context_statements = context_statements[:-1]

        self.context_statements = context_statements
        self.featureComputer = FeatureComputer(
            lang=self.lang,
            context_statements=context_statements,
            context_ids=input_ids,
            corpus=self.corpus,
            k=10
        )
        
        current_generate_len = 0
        current_input_ids = input_ids.to(self.device)
        l_model_pkv = None
        s_model_pkv = None

        with torch.no_grad():
            while current_generate_len < max_gen_len-1:
                if self.with_classifier:
                    output = self.csd_step(
                        current_input_ids=current_input_ids,
                        current_generate_len=current_generate_len,
                        prefix_ids=generate_ids[0, :current_generate_len],
                        l_model_pkv=l_model_pkv,
                        s_model_pkv=s_model_pkv,
                        max_draft_len=min(max_draft_len, max_gen_len-current_generate_len-1),
                        th_classifier_prob=th_classifier_prob,
                    )
                elif self.with_sd:
                    output = self.sd_step(
                        current_input_ids=current_input_ids,
                        l_model_pkv=l_model_pkv,
                        s_model_pkv=s_model_pkv,
                        max_draft_len=min(max_draft_len, max_gen_len-current_generate_len-1)
                    )

                output_ids = output['generate_ids']
                current_input_ids = output['next_input_ids']

                s_model_pkv = output['s_model_pkv']
                l_model_pkv = output['l_model_pkv']

                generate_ids[:, current_generate_len:current_generate_len+output_ids.size(1)] = output_ids
                current_generate_len += output_ids.size(1)
                
                if current_input_ids.item() == self.tokenizer.pad_token_id or current_input_ids.item() == self.tokenizer.eos_token_id:
                    break
                elif self.lang == 'python' and self.tokenizer.decode(current_input_ids.item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(current_input_ids.item()).strip() in ['{', ';', '}']:
                    break
            
        generate_ids = generate_ids[:, :current_generate_len]
        pred = self.tokenizer.decode(generate_ids[0]).strip()
        if '\n' in pred:
            pred = pred[:pred.find('\n')]
        if '<｜end▁of▁sentence｜' in pred:
            pred = pred[:pred.find('<｜end▁of▁sentence｜')]
    
        return pred.strip()
    
    def csd_step(self, current_input_ids, current_generate_len, prefix_ids, l_model_pkv, s_model_pkv, max_draft_len, th_classifier_prob):
        s_token_ids = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.long, device=self.device)
        s_token_probs = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.float, device=self.device)

        s_model_embeddings = []
        if self.s_model_type == 'deepseek-coder':
            s_model_all_token_probs = torch.empty([current_input_ids.size(0), max_draft_len, self.vocabulary_size], dtype=torch.float, device=self.device)
        # elif self.s_model_type == 'starcoder2':
        #     s_model_all_token_probs = torch.empty([current_input_ids.size(0), max_draft_len, self.vocabulary_size], dtype=torch.float, device=self.device)

        # generate by small model
        it = 0
        s_model_input_ids = current_input_ids.to(self.device)
        for it in range(max_draft_len):
            s_model_output = self.generate(
                model=self.s_model,
                model_type=self.s_model_type,
                past_key_values=s_model_pkv,
                input_ids=s_model_input_ids
            )
            
            s_token_ids[:, it] = s_model_output['ids'][:, -1]
            s_model_input_ids = s_model_output['ids'][:, -1:]

            s_token_probs[:, it] = s_model_output['probs'][:, -1]
            s_model_all_token_probs[:, it] = s_model_output['all_token_probs'][:, -1]

            s_model_pkv = s_model_output['past_key_values']
            
            s_model_embeddings.append(self.collector.s_model_ll_embedding[:, -1])

        draft_len = it+1
        s_token_ids = s_token_ids[:, :draft_len]
        s_token_probs = s_token_probs[:, :draft_len]

        # generate by large model
        l_model_input_ids = torch.cat((current_input_ids, s_token_ids), dim=-1).to(self.device)
        l_model_output = self.generate(
            model=self.l_model,
            model_type=self.l_model_type,
            past_key_values=l_model_pkv,
            input_ids=l_model_input_ids
        )

        target_len = draft_len+1
        l_token_ids = l_model_output['ids'][:, -target_len:]
        l_token_probs = l_model_output['probs'][:, -target_len:]
        l_model_all_token_probs = l_model_output['all_token_probs'][:, -target_len:]
        l_model_pkv = l_model_output['past_key_values']

        matched, generate_ids = self.verify(
            prefix_ids=prefix_ids,
            current_generate_len=current_generate_len,
            draft_len=draft_len,
            target_len=target_len,
            s_token_ids=s_token_ids,
            l_token_ids=l_token_ids,
            s_token_probs=s_token_probs,
            l_token_probs=l_token_probs,
            s_model_all_token_probs=s_model_all_token_probs,
            l_model_all_token_probs=l_model_all_token_probs,
            s_model_embeddings=s_model_embeddings,
            th_classifier_prob=th_classifier_prob
        )
        
        # set pkv
        if matched < draft_len:
            s_model_pkv = [
                (k[:, :, :-(draft_len-matched-1)], v[:, :, :-(draft_len-matched-1)]) for k, v in s_model_pkv
            ]
            l_model_pkv = [
                (k[:, :, :-(target_len-matched-1)], v[:, :, :-(target_len-matched-1)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]
        else:
            l_model_pkv = [
                (k[:, :, :-(target_len-matched)], v[:, :, :-(target_len-matched)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]

        return {
            'generate_ids': generate_ids,
            'next_input_ids': next_input_ids,
            's_model_pkv': s_model_pkv,
            'l_model_pkv': l_model_pkv
        }
    
    def verify(self, prefix_ids, current_generate_len, draft_len, target_len, s_token_ids, l_token_ids, s_token_probs, l_token_probs, s_model_all_token_probs, l_model_all_token_probs, s_model_embeddings, th_classifier_prob):
        generate_ids = torch.empty([1, target_len], dtype=torch.long, device=self.device)
        generate_len = 0
        
        matched = 0
        l_model_embeddings = self.collector.l_model_ll_embedding # [input_ids.shape[0], target_len, 4096]
        for i in range(draft_len):
            if s_token_ids[:, i] == l_token_ids[:, i]:
                matched += 1
                generate_ids[:, i] = s_token_ids[:, i]
                generate_len += 1
                if self.lang == 'python' and self.tokenizer.decode(generate_ids[:, i].item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(generate_ids[:, i].item()).strip() in ['{', ';', '}']:
                    break
                
            else:
                # token_id, prob
                l_token_id = l_token_ids[0, i].to(self.device)
                s_token_id = s_token_ids[0, i].to(self.device)

                l_token_prob = l_token_probs[0, i]
                s_token_prob = s_token_probs[0, i]
                l_prob_of_s_token = l_model_all_token_probs[0, i][s_token_id]
                s_prob_of_l_token = s_model_all_token_probs[0, i][l_token_id]
                
                # embedding
                embedding = torch.cat(
                    (
                        l_model_embeddings[:, -(target_len-i)],
                        s_model_embeddings[i]
                    ), dim=-1
                )

                # Type
                l_word = self.tokenizer.decode(l_token_id)
                s_word = self.tokenizer.decode(s_token_id)
                l_word_type = self.featureComputer.get_word_type(l_word)
                s_word_type = self.featureComputer.get_word_type(s_word)

                # Frequency
                l_token_id_cnt = self.featureComputer.get_token_id_cnt(l_token_id)
                s_token_id_cnt = self.featureComputer.get_token_id_cnt(s_token_id)                
                l_word = l_word.strip()
                s_word = s_word.strip()
                l_word_cnt = self.featureComputer.get_word_cnt(l_word)
                s_word_cnt = self.featureComputer.get_word_cnt(s_word)
                l_subword_cnt = self.featureComputer.get_subword_cnt(l_word)
                s_subword_cnt = self.featureComputer.get_subword_cnt(s_word)
                l_word_gauss_cnt = self.featureComputer.get_word_gauss_cnt(l_word)
                s_word_gauss_cnt = self.featureComputer.get_word_gauss_cnt(s_word)
                l_subword_gauss_cnt = self.featureComputer.get_subword_gauss_cnt(l_word)
                s_subword_gauss_cnt = self.featureComputer.get_subword_gauss_cnt(s_word)

                l_statement_ids = torch.cat((prefix_ids, s_token_ids[0, :i], l_token_ids[0, i:i+1]), dim=-1)
                s_statement_ids = torch.cat((prefix_ids, s_token_ids[0, :i+1]), dim=-1)
                l_statement_ids_cnt = self.featureComputer.get_statement_ids_cnt(l_statement_ids)
                s_statement_ids_cnt = self.featureComputer.get_statement_ids_cnt(s_statement_ids)

                l_statement = self.tokenizer.decode(l_statement_ids).strip()
                s_statement = self.tokenizer.decode(s_statement_ids).strip()
                if not self.nextline:
                    l_statement = self.gt_prefix + l_statement
                    s_statement = self.gt_prefix + s_statement
                l_statement_cnt = self.featureComputer.get_statement_cnt(l_statement)
                s_statement_cnt = self.featureComputer.get_statement_cnt(s_statement)

                # Similarity
                l_words = l_statement.split()
                s_words = s_statement.split()
                l_bm25_of_context = self.featureComputer.get_bm25_of_context(l_words)
                s_bm25_of_context = self.featureComputer.get_bm25_of_context(s_words)
                l_bm25_of_lastline = self.featureComputer.get_bm25_of_lastline(l_words)
                s_bm25_of_lastline = self.featureComputer.get_bm25_of_lastline(s_words)
                l_bm25_of_context_with_lastline = self.featureComputer.get_bm25_of_context_with_lastline(l_words)
                s_bm25_of_context_with_lastline = self.featureComputer.get_bm25_of_context_with_lastline(s_words)

                lastline = self.context_statements[-1]
                # Frequency
                l_multiline_cnt = self.featureComputer.get_multiline_cnt(lastline, l_statement)
                s_multiline_cnt = self.featureComputer.get_multiline_cnt(lastline, s_statement)

                # Retrieval
                l_statement_cnt_in_corpus, s_statement_cnt_in_corpus, l_statement_cnt_in_corpus_with_lastline, s_statement_cnt_in_corpus_with_lastline = self.featureComputer.get_cnt_in_corpus(
                    lastline=lastline,
                    l_statement=l_statement,
                    s_statement=s_statement
                )
                l_avg_top_k_jaccard_with_lastline_in_corpus, s_avg_top_k_jaccard_with_lastline_in_corpus, l_top_k_jaccard_in_corpus, s_top_k_jaccard_in_corpus = self.featureComputer.get_jaccard_in_corpus(
                    lastline=lastline,
                    l_words=l_words,
                    s_words=s_words
                )

                # Length
                len_input_ids = self.featureComputer.get_len_input_ids()
                len_input_words = self.featureComputer.get_len_input_words()
                len_generate_token_ids = current_generate_len+i+1
                len_generate_words = min(len(l_words), len(s_words))
                len_statements = self.featureComputer.get_len_statements()
                len_lastline = self.featureComputer.get_len_lastline()

                feature = {
                    # Length
                    'len_generate_token_ids': torch.tensor(len_generate_token_ids).to(self.device),
                    'len_generate_words': torch.tensor(len_generate_words).to(self.device),
                    'len_input_ids': torch.tensor(len_input_ids).to(self.device),
                    'len_input_words': torch.tensor(len_input_words).to(self.device),
                    'len_statements': torch.tensor(len_statements).to(self.device),
                    'len_lastline': torch.tensor(len_lastline).to(self.device),

                    # Prob
                    'l_token_prob': l_token_prob,
                    's_token_prob': s_token_prob,
                    'l_prob_of_s_token': l_prob_of_s_token,
                    's_prob_of_l_token': s_prob_of_l_token,

                    # Type
                    'l_word_type': torch.tensor(l_word_type).to(self.device),
                    's_word_type': torch.tensor(s_word_type).to(self.device),

                    # Frequency
                    'l_token_id_cnt': l_token_id_cnt,
                    's_token_id_cnt': s_token_id_cnt,
                    'l_word_cnt': torch.tensor(l_word_cnt).to(self.device),
                    's_word_cnt': torch.tensor(s_word_cnt).to(self.device),
                    'l_subword_cnt': torch.tensor(l_subword_cnt).to(self.device),
                    's_subword_cnt': torch.tensor(s_subword_cnt).to(self.device),
                    'l_word_gauss_cnt': torch.tensor(l_word_gauss_cnt).to(self.device),
                    's_word_gauss_cnt': torch.tensor(s_word_gauss_cnt).to(self.device),
                    'l_subword_gauss_cnt': torch.tensor(l_subword_gauss_cnt).to(self.device),
                    's_subword_gauss_cnt': torch.tensor(s_subword_gauss_cnt).to(self.device),
                    'l_statement_ids_cnt': torch.tensor(l_statement_ids_cnt).to(self.device),
                    's_statement_ids_cnt': torch.tensor(s_statement_ids_cnt).to(self.device),
                    'l_statement_cnt': torch.tensor(l_statement_cnt).to(self.device),
                    's_statement_cnt': torch.tensor(s_statement_cnt).to(self.device),
                    'l_multiline_cnt': torch.tensor(l_multiline_cnt).to(self.device),
                    's_multiline_cnt': torch.tensor(s_multiline_cnt).to(self.device),

                    # Similarity
                    'l_bm25_of_context': torch.tensor(l_bm25_of_context).to(self.device),
                    's_bm25_of_context': torch.tensor(s_bm25_of_context).to(self.device),
                    'l_bm25_of_lastline': torch.tensor(l_bm25_of_lastline).to(self.device),
                    's_bm25_of_lastline': torch.tensor(s_bm25_of_lastline).to(self.device),
                    'l_bm25_of_context_with_lastline': torch.tensor(l_bm25_of_context_with_lastline).to(self.device),
                    's_bm25_of_context_with_lastline': torch.tensor(s_bm25_of_context_with_lastline).to(self.device),

                    # Retrieval
                    'l_statement_cnt_in_corpus': torch.tensor(l_statement_cnt_in_corpus).to(self.device),
                    's_statement_cnt_in_corpus': torch.tensor(s_statement_cnt_in_corpus).to(self.device),
                    'l_statement_cnt_in_corpus_with_lastline': torch.tensor(l_statement_cnt_in_corpus_with_lastline).to(self.device),
                    's_statement_cnt_in_corpus_with_lastline': torch.tensor(s_statement_cnt_in_corpus_with_lastline).to(self.device),

                    'l_avg_top_k_jaccard_with_lastline_in_corpus': torch.tensor(l_avg_top_k_jaccard_with_lastline_in_corpus).to(self.device),
                    's_avg_top_k_jaccard_with_lastline_in_corpus': torch.tensor(s_avg_top_k_jaccard_with_lastline_in_corpus).to(self.device),
                    'l_top_k_jaccard_in_corpus': torch.tensor(l_top_k_jaccard_in_corpus).to(self.device),
                    's_top_k_jaccard_in_corpus': torch.tensor(s_top_k_jaccard_in_corpus).to(self.device),
                }

                for k, v in feature.items():
                    feature[k] = v.unsqueeze(0) 

                yhat = self.classifier.predict(embedding, feature)

                if yhat > th_classifier_prob:  # trust small model
                    matched += 1
                    generate_ids[:, i] = s_token_ids[:, i]
                    generate_len += 1
                    if self.lang == 'python' and self.tokenizer.decode(generate_ids[:, i].item()) == '\n':
                        break
                    elif self.lang == 'java' and self.tokenizer.decode(generate_ids[:, i].item()).strip() in ['{', ';', '}']:
                        break
                else:   # trust large model
                    generate_ids[:, i] = l_token_ids[:, i]
                    generate_len += 1
                    break

        if matched == draft_len:
            generate_ids[:, matched] = l_token_ids[:, matched]
        
        return matched, generate_ids[:, :generate_len]

    # def sd_step(self, current_input_ids, l_model_pkv, s_model_pkv, max_draft_len):
    #     t = time.perf_counter()
    #     s_token_ids = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.long, device=self.device)

    #     it = 0
    #     s_model_input_ids = current_input_ids.to(self.device)
    #     for it in range(max_draft_len):
    #         s_model_output = self.generate(
    #             model=self.s_model,
    #             model_type=self.s_model_type,
    #             past_key_values=s_model_pkv,
    #             input_ids=s_model_input_ids
    #         )
            
    #         s_token_ids[:, it] = s_model_output['ids'][:, -1]
    #         s_model_input_ids = s_model_output['ids'][:, -1:]
    #         s_model_pkv = s_model_output['past_key_values']

    #     draft_len = it+1
    #     s_token_ids = s_token_ids[:, :draft_len]
    #     print(f'small model\'s sampling cost: {time.perf_counter() - t:.8f}s')
    #     t = time.perf_counter()

    #     l_model_input_ids = torch.cat((current_input_ids, s_token_ids), dim=-1).to(self.device)
    #     l_model_output = self.generate(
    #         model=self.l_model,
    #         model_type=self.l_model_type,
    #         past_key_values=l_model_pkv,
    #         input_ids=l_model_input_ids
    #     )

    #     target_len = draft_len+1
    #     l_token_ids = l_model_output['ids'][:, -target_len:]
    #     l_model_pkv = l_model_output['past_key_values']
    #     print(f'large model\'s sampling cost: {time.perf_counter() - t:.8f}s')
    #     t = time.perf_counter()

    #     print(f's_token_ids = {s_token_ids}, l_token_ids = {l_token_ids}')
    #     matched = ((s_token_ids[:, :] != l_token_ids[:, :-1]).cumsum(-1) == 0).sum(-1).item()
    #     print(f'compute matched cost: {time.perf_counter() - t:.8f}s, matched = {matched}')
    #     t = time.perf_counter()
    #     generate_ids = torch.cat((s_token_ids[:, :matched], l_token_ids[:, matched:matched+1]), dim=-1)
        
    #     # set pkv
    #     if matched < draft_len:
    #         s_model_pkv = [
    #             (k[:, :, :-(draft_len-matched-1)], v[:, :, :-(draft_len-matched-1)]) for k, v in s_model_pkv
    #         ]
    #         l_model_pkv = [
    #             (k[:, :, :-(target_len-matched-1)], v[:, :, :-(target_len-matched-1)]) for k, v in l_model_pkv
    #         ]
    #         next_input_ids = generate_ids[:, -1:]
    #     else:
    #         l_model_pkv = [
    #             (k[:, :, :-(target_len-matched)], v[:, :, :-(target_len-matched)]) for k, v in l_model_pkv
    #         ]
    #         next_input_ids = generate_ids[:, -2:]
    #     print(f'update pkv cost: {time.perf_counter() - t:.8f}s\n')

    #     return {
    #         'generate_ids': generate_ids,
    #         'next_input_ids': next_input_ids,
    #         's_model_pkv': s_model_pkv,
    #         'l_model_pkv': l_model_pkv
    #     }

class Bild_Generator:
    def __init__(self, l_model, l_model_type, tokenizer, device, lang, s_model, s_model_type, fallback_th, rollback_th):
        self.l_model = l_model
        self.l_model_type = l_model_type
        self.s_model = s_model
        self.s_model_type = s_model_type

        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
            
        if fallback_th is not None and rollback_th is not None:
            self.with_fallback_rollback = True
            self.fallback_th = fallback_th
            self.rollback_th = rollback_th
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduce=False)
        else:
            self.with_fallback_rollback = False

    def generate(self, model, model_type, past_key_values, input_ids):
        if model_type in ['deepseek-coder', 'starcoder2']:
            input_ids = input_ids.to(self.device)
            output = model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True
                        )
            logits = output.logits
            past_key_values = output.past_key_values

            probs = torch.nn.functional.softmax(logits, dim=-1)

            output_probs, output_ids = torch.max(probs, dim=-1)
        else:
            raise NotImplementedError('Unknown model type')
        
        return {
            'ids': output_ids,
            'probs': output_probs,
            'logits': logits,
            'past_key_values': past_key_values
        }


    def sd_bild(self, input_ids, max_gen_len, max_draft_len):
        generate_ids = torch.empty([input_ids.size(0), max_gen_len], dtype=torch.long, device=self.device)

        current_generate_len = 0
        current_input_ids = input_ids.to(self.device)
        l_model_pkv = None
        s_model_pkv = None

        with torch.no_grad():
            while current_generate_len < max_gen_len-1:
                output = self.sd_bild_step(
                    current_input_ids=current_input_ids,
                    l_model_pkv=l_model_pkv,
                    s_model_pkv=s_model_pkv,
                    max_draft_len=min(max_draft_len, max_gen_len-current_generate_len-1)
                )

                output_ids = output['generate_ids']
                current_input_ids = output['next_input_ids']

                s_model_pkv = output['s_model_pkv']
                l_model_pkv = output['l_model_pkv']

                generate_ids[:, current_generate_len:current_generate_len+output_ids.size(1)] = output_ids
                current_generate_len += output_ids.size(1)
                
                if current_input_ids.item() == self.tokenizer.pad_token_id or current_input_ids.item() == self.tokenizer.eos_token_id:
                    break
                elif self.lang == 'python' and self.tokenizer.decode(current_input_ids.item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(current_input_ids.item()).strip() in ['{', ';', '}']:
                    break
        
        generate_ids = generate_ids[:, :current_generate_len]
        pred = self.tokenizer.decode(generate_ids[0]).strip()
        if '\n' in pred:
            pred = pred[:pred.find('\n')]
        if '<｜end▁of▁sentence｜' in pred:
            pred = pred[:pred.find('<｜end▁of▁sentence｜')]
    
        return pred.strip()


    def sd_bild_step(self, current_input_ids, l_model_pkv, s_model_pkv, max_draft_len):
        s_token_ids = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.long, device=self.device)
        s_token_probs = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.float, device=self.device)

        it = 0
        s_model_input_ids = current_input_ids.to(self.device)

        for it in range(max_draft_len):
            s_model_output = self.generate(
                model=self.s_model,
                model_type=self.s_model_type,
                past_key_values=s_model_pkv,
                input_ids=s_model_input_ids
            )

            s_token_ids[:, it] = s_model_output['ids'][:, -1]
            s_model_input_ids = s_model_output['ids'][:, -1:]

            s_token_probs[:, it] = s_model_output['probs'][:, -1]

            s_model_pkv = s_model_output['past_key_values']

            if self.with_fallback_rollback and s_token_probs[:, it] < self.fallback_th:
                break

        draft_len = it+1
        s_token_ids = s_token_ids[:, :draft_len]
        s_token_probs = s_token_probs[:, :draft_len]

        l_model_input_ids = torch.cat((current_input_ids, s_token_ids), dim=-1).to(self.device)
        l_model_output = self.generate(
            model=self.l_model,
            model_type=self.l_model_type,
            past_key_values=l_model_pkv,
            input_ids=l_model_input_ids
        )

        target_len = draft_len+1
        l_token_ids = l_model_output['ids'][:, -target_len:]
        l_token_probs = l_model_output['probs'][:, -target_len:]
        l_model_pkv = l_model_output['past_key_values']
        l_model_logits = l_model_output['logits'][:, -target_len:]

        matched, generate_ids = self.sd_bild_verify(
            draft_len=draft_len,
            target_len=target_len,
            s_token_ids=s_token_ids,
            l_token_ids=l_token_ids,
            s_token_probs=s_token_probs,
            l_token_probs=l_token_probs,
            l_model_logits=l_model_logits
        )

        if matched < draft_len:
            s_model_pkv = [
                (k[:, :, :-(draft_len-matched-1)], v[:, :, :-(draft_len-matched-1)]) for k, v in s_model_pkv
            ]
            l_model_pkv = [
                (k[:, :, :-(target_len-matched-1)], v[:, :, :-(target_len-matched-1)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]
        else:
            l_model_pkv = [
                (k[:, :, :-(target_len-matched)], v[:, :, :-(target_len-matched)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]

        return {
            'generate_ids': generate_ids,
            'next_input_ids': next_input_ids,
            's_model_pkv': s_model_pkv,
            'l_model_pkv': l_model_pkv
        }
    
    def sd_bild_verify(self, draft_len, target_len, s_token_ids, l_token_ids, s_token_probs, l_token_probs, l_model_logits):

        generate_ids = torch.empty([1, target_len], dtype=torch.long, device=self.device)
        generate_len = 0
        
        matched = 0
        
        for i in range(draft_len):
            if s_token_ids[:, i] == l_token_ids[:, i]:
                matched += 1
                generate_ids[:, i] = s_token_ids[:, i]
                generate_len += 1

                if self.lang == 'python' and self.tokenizer.decode(generate_ids[:, i].item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(generate_ids[:, i].item()).strip() in ['{', ';', '}']:
                    break
            
            else:
                s_token_id = s_token_ids[0, i].to(self.device)

                l_token_prob = l_token_probs[0, i]
                s_token_prob = s_token_probs[0, i]

                l_model_logit = l_model_logits[0, i]

                if self.with_fallback_rollback:
                    label = self.cross_entropy_loss(l_model_logit, s_token_id) < self.rollback_th
                else:
                    label = random.random() < s_token_prob / l_token_prob

                if label:
                    matched += 1
                    generate_ids[:, i] = s_token_ids[:, i]
                    generate_len += 1
                    
                    if self.lang == 'python' and self.tokenizer.decode(generate_ids[:, i].item()) == '\n':
                        break
                    elif self.lang == 'java' and self.tokenizer.decode(generate_ids[:, i].item()).strip() in ['{', ';', '}']:
                        break
                else:
                    generate_ids[:, i] = l_token_ids[:, i]
                    generate_len += 1
                    break
        
        if matched == draft_len:
            generate_ids[:, matched] = l_token_ids[:, matched]

        return matched, generate_ids[:, :generate_len]
    


class Raw_SD_Generator:
    def __init__(self, l_model, l_model_type, tokenizer, device, lang, s_model, s_model_type):
        self.l_model = l_model
        self.l_model_type = l_model_type
        self.s_model = s_model
        self.s_model_type = s_model_type

        self.tokenizer = tokenizer
        self.device = device
        self.lang = lang
            
    def generate(self, model, model_type, past_key_values, input_ids):
        if model_type in ['deepseek-coder', 'starcoder2']:
            input_ids = input_ids.to(self.device)
            output = model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        return_dict=True,
                        use_cache=True
                        )
            logits = output.logits
            past_key_values = output.past_key_values

            probs = torch.nn.functional.softmax(logits, dim=-1)

            output_probs, output_ids = torch.max(probs, dim=-1)
        else:
            raise NotImplementedError('Unknown model type')
        
        return {
            'ids': output_ids,
            'probs': output_probs,
            'all_token_probs': probs,
            'logits': logits,
            'past_key_values': past_key_values
        }


    def speculative_decoding(self, input_ids, max_gen_len, max_draft_len):
        generate_ids = torch.empty([input_ids.size(0), max_gen_len], dtype=torch.long, device=self.device)

        current_generate_len = 0
        current_input_ids = input_ids.to(self.device)
        l_model_pkv = None
        s_model_pkv = None

        with torch.no_grad():
            while current_generate_len < max_gen_len-1:
                output = self.sd_step(
                    current_input_ids=current_input_ids,
                    l_model_pkv=l_model_pkv,
                    s_model_pkv=s_model_pkv,
                    max_draft_len=min(max_draft_len, max_gen_len-current_generate_len-1)
                )

                output_ids = output['generate_ids']
                current_input_ids = output['next_input_ids']

                s_model_pkv = output['s_model_pkv']
                l_model_pkv = output['l_model_pkv']

                generate_ids[:, current_generate_len:current_generate_len+output_ids.size(1)] = output_ids
                current_generate_len += output_ids.size(1)
                
                if current_input_ids.item() == self.tokenizer.pad_token_id or current_input_ids.item() == self.tokenizer.eos_token_id:
                    break
                elif self.lang == 'python' and self.tokenizer.decode(current_input_ids.item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(current_input_ids.item()).strip() in ['{', ';', '}']:
                    break
        
        generate_ids = generate_ids[:, :current_generate_len]
        pred = self.tokenizer.decode(generate_ids[0]).strip()
        if '\n' in pred:
            pred = pred[:pred.find('\n')]
        if '<｜end▁of▁sentence｜' in pred:
            pred = pred[:pred.find('<｜end▁of▁sentence｜')]
    
        return pred.strip()


    def sd_step(self, current_input_ids, l_model_pkv, s_model_pkv, max_draft_len):
        
        s_token_ids = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.long, device=self.device)
        s_token_probs = torch.empty([current_input_ids.size(0), max_draft_len], dtype=torch.float, device=self.device)
        s_all_token_probs = torch.empty([current_input_ids.size(0), max_draft_len, 32256], dtype=torch.float, device=self.device)

        it = 0
        s_model_input_ids = current_input_ids.to(self.device)

        for it in range(max_draft_len):
            s_model_output = self.generate(
                model=self.s_model,
                model_type=self.s_model_type,
                past_key_values=s_model_pkv,
                input_ids=s_model_input_ids
            )

            s_token_ids[:, it] = s_model_output['ids'][:, -1]
            s_model_input_ids = s_model_output['ids'][:, -1:]

            s_token_probs[:, it] = s_model_output['probs'][:, -1]
            s_all_token_probs[:, it] = s_model_output['all_token_probs'][:, -1]

            s_model_pkv = s_model_output['past_key_values']


        draft_len = it+1
        s_token_ids = s_token_ids[:, :draft_len]
        s_token_probs = s_token_probs[:, :draft_len]

        l_model_input_ids = torch.cat((current_input_ids, s_token_ids), dim=-1).to(self.device)
        l_model_output = self.generate(
            model=self.l_model,
            model_type=self.l_model_type,
            past_key_values=l_model_pkv,
            input_ids=l_model_input_ids
        )

        target_len = draft_len+1
        l_token_ids = l_model_output['ids'][:, -target_len:]
        l_all_token_probs = l_model_output['all_token_probs'][:, -target_len:]
        l_model_pkv = l_model_output['past_key_values']

        matched, generate_ids = self.verify(
            draft_len=draft_len,
            target_len=target_len,
            s_token_ids=s_token_ids,
            l_token_ids=l_token_ids,
            s_token_probs=s_token_probs,
            s_all_token_probs=s_all_token_probs,
            l_all_token_probs=l_all_token_probs
        )

        if matched < draft_len:
            s_model_pkv = [
                (k[:, :, :-(draft_len-matched-1)], v[:, :, :-(draft_len-matched-1)]) for k, v in s_model_pkv
            ]
            l_model_pkv = [
                (k[:, :, :-(target_len-matched-1)], v[:, :, :-(target_len-matched-1)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]
        else:
            l_model_pkv = [
                (k[:, :, :-(target_len-matched)], v[:, :, :-(target_len-matched)]) for k, v in l_model_pkv
            ]
            next_input_ids = generate_ids[:, -1:]

        return {
            'generate_ids': generate_ids,
            'next_input_ids': next_input_ids,
            's_model_pkv': s_model_pkv,
            'l_model_pkv': l_model_pkv
        }
    
    def verify(self, draft_len, target_len, s_token_ids, l_token_ids, s_token_probs, s_all_token_probs, l_all_token_probs):

        generate_ids = torch.empty([1, target_len], dtype=torch.long, device=self.device)
        generate_len = 0
        
        matched = 0
        
        for i in range(draft_len):
            token_id = s_token_ids[0, i].to(self.device)
            s_token_prob = s_token_probs[0, i]
            l_token_prob = l_all_token_probs[0, i][token_id]

            label = random.random() < l_token_prob / s_token_prob

            if label:
                matched += 1
                generate_ids[:, i] = token_id
                generate_len += 1

                if self.lang == 'python' and self.tokenizer.decode(generate_ids[:, i].item()) == '\n':
                    break
                elif self.lang == 'java' and self.tokenizer.decode(generate_ids[:, i].item()).strip() in ['{', ';', '}']:
                    break
            
            else:
                l_all_current_token_probs = l_all_token_probs[0, i]
                s_all_current_token_probs = s_all_token_probs[0, i]
                
                new_token_probs = l_all_current_token_probs - s_all_current_token_probs
                new_token_probs = torch.nn.functional.relu(new_token_probs, inplace=True)
                new_token_probs = new_token_probs / new_token_probs.sum()

                new_sampled_id = torch.multinomial(new_token_probs, num_samples=1)
                
                generate_ids[:, i] = new_sampled_id.squeeze(-1)
                generate_len += 1
                break    

        return matched, generate_ids[:, :generate_len]