import math
import random
import re

from collections import Counter

import keyword
import string

import torch
import numpy as np
import torch.backends
import torch.backends.cudnn

def set_seed(param):
    random.seed(param)
    np.random.seed(param)
    torch.manual_seed(param)
    torch.cuda.manual_seed_all(param)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_py_code(code):
    return code.strip().split('\n')

def split_java_code(code):
    tokens = code.split()
    statements = []
    cur_tokens = []
    for token in tokens:
        cur_tokens.append(token)
        if token in [';', '{', '}']:
            statements.append(" ".join(cur_tokens))
            cur_tokens = []

    if len(cur_tokens) > 0:
        statements.append(" ".join(cur_tokens))

    return statements


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]
        self.avgdl = sum(self.doc_len) / len(docs)
        self.doc_freqs = []
        self.idf = {}
        self.initialize()

    def initialize(self):
        df = {} 
        for doc in self.docs:
            self.doc_freqs.append(Counter(doc))
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
                
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.docs) - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, doc, query):
        score = 0.0
        for word in query:
            if word in self.doc_freqs[doc]:
                freq = self.doc_freqs[doc][word]
                
                score += (self.idf[word] * freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl))
        return score

class CorpusRetriever:
    def __init__(self, corpus, lang, k):
        self.lang = lang
        self.k = k

        self.docs = []
        self.docs_set = []
        if lang == 'python':
            for code in corpus:
                code = code.strip()[3:-4].strip().replace(' <EOL> ', '\n')   # remove <s>, </s>
                statements = split_py_code(code)
                self.docs.append(statements)
                
                tmp_list = []
                for statement in statements:
                    tokens = statement.split()
                    tmp_list.append(set(tokens))
                self.docs_set.append(tmp_list)

        elif lang == 'java':
            for code in corpus:
                code = code.strip()[3:-4].strip()   # remove <s>, </s>
                statements = split_java_code(code)
                self.docs.append(statements)

                tmp_list = []
                for statement in statements:
                    tokens = statement.split()
                    tmp_list.append(set(tokens))
                self.docs_set.append(tmp_list)


    def get_cnt_feature(self, lastline, l_statement, s_statement):
        l_statement_cnt_in_corpus = 0
        s_statement_cnt_in_corpus = 0
        l_statement_cnt_in_corpus_with_lastline = 0
        s_statement_cnt_in_corpus_with_lastline = 0
        
        for statements in self.docs:
            for it, statement in enumerate(statements):
                l_statement_cnt_in_corpus += 1 if statement.startswith(l_statement) else 0
                s_statement_cnt_in_corpus += 1 if statement.startswith(s_statement) else 0

                if lastline == statement and it != len(statements) - 1:
                    l_statement_cnt_in_corpus_with_lastline += 1 if statements[it+1].startswith(l_statement) else 0
                    s_statement_cnt_in_corpus_with_lastline += 1 if statements[it+1].startswith(s_statement) else 0

        return l_statement_cnt_in_corpus, s_statement_cnt_in_corpus, l_statement_cnt_in_corpus_with_lastline, s_statement_cnt_in_corpus_with_lastline

    def get_jaccard_feature(self, lastline_words, l_words, s_words):
        l_query = set(l_words)
        s_query = set(s_words)
        lastline_query = set(lastline_words)

        l_scores = []
        s_scores = []
        l_scores_with_lastline = []
        s_scores_with_lastline = []

        for code in self.docs_set:
            l_temp_scores = []  
            s_temp_scores = []
            lastline_temp_scores = []

            for statement in code:
                l_temp_scores.append(len(l_query & statement) / len(l_query | statement))
                s_temp_scores.append(len(s_query & statement) / len(s_query | statement))
                lastline_temp_scores.append(len(lastline_query & statement) / len(lastline_query | statement))

            l_scores_with_lastline.extend(list(zip(lastline_temp_scores, l_temp_scores[1:])))
            s_scores_with_lastline.extend(list(zip(lastline_temp_scores, s_temp_scores[1:])))

            l_scores.extend(l_temp_scores)
            s_scores.extend(s_temp_scores)
            
        l_top_k_scores_with_lastline = [pair[1] for pair in sorted(l_scores_with_lastline, key=lambda x: x[0], reverse=True)[:self.k]]
        s_top_k_scores_with_lastline = [pair[1] for pair in sorted(s_scores_with_lastline, key=lambda x: x[0], reverse=True)[:self.k]]
        l_avg_top_k_jaccard_with_lastline_in_corpus = sum(l_top_k_scores_with_lastline) / len(l_top_k_scores_with_lastline)
        s_avg_top_k_jaccard_with_lastline_in_corpus = sum(s_top_k_scores_with_lastline) / len(s_top_k_scores_with_lastline)

        l_scores = sorted(l_scores, reverse=True)
        s_scores = sorted(s_scores, reverse=True)
        l_top_k_jaccard_in_corpus = l_scores[self.k-1]
        s_top_k_jaccard_in_corpus = s_scores[self.k-1]

        return l_avg_top_k_jaccard_with_lastline_in_corpus, s_avg_top_k_jaccard_with_lastline_in_corpus, l_top_k_jaccard_in_corpus, s_top_k_jaccard_in_corpus

class FeatureComputer:
    def __init__(self, lang, context_statements, context_ids, corpus, k):
        self.lang = lang    # python or java

        self.context_statements = context_statements
        self.context_docs = [s.split(" ") for s in context_statements]
        self.context_bm25 = BM25(self.context_docs)

        self.context_ids = context_ids  # [bs, len(input_ids)]
        self.device = self.context_ids[0].device

        self.corpusRetriever = CorpusRetriever(corpus, lang, k)

        self.feature_len_input_ids = self.context_ids.shape[1]
        self.feature_len_input_words = sum(len(doc) for doc in self.context_docs)
        self.feature_len_statements = len(context_statements)
        self.feature_len_lastline = len(context_statements[-1].split())

    def get_len_input_ids(self):
        return self.feature_len_input_ids
    
    def get_len_input_words(self):
        return self.feature_len_input_words
    
    def get_len_statements(self):
        return self.feature_len_statements
    
    def get_len_lastline(self):
        return self.feature_len_lastline

    def get_token_id_cnt(self, token_id):
        token_id = token_id.to(self.device)
        return torch.sum(self.context_ids[0].eq(token_id.item()))

    def get_word_type(self, str):
        if str == '\n':
            return 0
        elif str == ' ':
            return 1
        
        def is_numeric(s):
            return s.isnumeric()

        def is_python_keyword(s):
            return keyword.iskeyword(s)
        
        def is_java_keyword(s):
            return s in ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'continue', 'default', 'do', 'double', 'else', 'enum', 'exports', 'extends', 'final', 'finally', 'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'module', 'native', 'new', 'package', 'private', 'protected', 'public', 'requires', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while']

        def is_punctuation_char(s):
            return all(ch in string.punctuation for ch in s)

        def is_valid_python_identifier(s):
            return s.isidentifier()
        
        def is_valid_java_identifier(s):
            pattern = r'^[a-zA-Z_$][a-zA-Z0-9_$]*$'
            return bool(re.match(pattern, s))

        str = str.strip()
        if str == '':   # e.g. ĠĠĠ
            return 1
        elif is_punctuation_char(str):
            return 2
        elif (self.lang == 'python' and is_python_keyword(str)) or (self.lang == 'java' and is_java_keyword(str)):
            return 3
        elif (self.lang == 'python' and is_valid_python_identifier(str)) or (self.lang == 'java' and is_valid_java_identifier(str)):
            return 4
        elif is_numeric(str):
            return 5
        else:
            return 6    # e.g. <EOS>
        
    def get_word_cnt(self, str):
        return sum(str == word for doc in self.context_docs for word in doc)

    def get_subword_cnt(self, str):
        return sum(str in word for doc in self.context_docs for word in doc)
    
    def gauss_decay(self, distance, scale):
        return math.exp(-0.5 * (distance / scale) ** 2)

    def get_word_gauss_cnt(self, str):
        total, scale, distance = 0, 2, 0
        for doc in self.context_docs[::-1]:
            count = sum(str == word for word in doc)
            if count > 0:
                total += count * self.gauss_decay(distance, scale)
            distance += 1
        return total

    def get_subword_gauss_cnt(self, str):
        total, scale, distance = 0, 2, 0
        for doc in self.context_docs[::-1]:
            count = sum(str in word for word in doc)
            if count > 0:
                total += count * self.gauss_decay(distance, scale)
            distance += 1
        return total
        
    def get_statement_ids_cnt(self, token_ids):
        token_ids = token_ids.to(self.device)
        count = 0
        for i in range(len(self.context_ids[0]) - len(token_ids) + 1):
            if torch.equal(self.context_ids[0][i:i+len(token_ids)], token_ids):
                count += 1
        return count

    def get_statement_cnt(self, str):
        return sum(str in statement for statement in self.context_statements)
        
    def get_multiline_cnt(self, lastline, str):
        ret = 0
        for i in range(len(self.context_statements)-1):
            if self.context_statements[i] == lastline and self.context_statements[i+1].startswith(str):
                ret += 1
        return ret

    def get_bm25_of_context(self, query):
        total, scale, distance = 0, 2, 0

        for di in range(len(self.context_docs)-1, -1, -1):
            score = self.context_bm25.score(di, query)
            if score > 0:
                total += score * self.gauss_decay(distance, scale)

        return total

    def get_bm25_of_lastline(self, query):
        return self.context_bm25.score(-1, query)

    def get_bm25_of_context_with_lastline(self, query):
        total, th = 0, 5

        last_line = self.context_docs[-1]
        for i in range(len(self.context_docs)):
            if i >= len(self.context_docs) - 1:
                break
            if self.context_bm25.score(i, last_line) > th:
                total += self.context_bm25.score(i+1, query)

        return total
    
    def get_cnt_in_corpus(self, lastline, l_statement, s_statement):
        return self.corpusRetriever.get_cnt_feature(
            lastline=lastline,
            l_statement=l_statement,
            s_statement=s_statement
        )
    
    def get_jaccard_in_corpus(self, lastline, l_words, s_words):
        return self.corpusRetriever.get_jaccard_feature(
            lastline_words=lastline.split(),
            l_words=l_words,
            s_words=s_words
        )
