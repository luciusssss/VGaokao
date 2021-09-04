import io, json, codecs, os
import random
ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)


import re
from tqdm import tqdm
import numpy as np
from pprint import pprint
import pickle as pk
import jieba 
from sentence_transformers import SentenceTransformer, util

# https://cloud.tencent.com/developer/article/1530340
def sent_tokenize(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    para = para.split("\n")
    ret = []
    for s in para:
        if len(s) > 0:
            ret.append(s)
    return ret


punctuations = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。.,/?![]\"\\"

simple_stop_list = [c for c in punctuations]
for line in open('../data/external/stop_words.txt', 'r', encoding='utf-8'):
    simple_stop_list.append(line.strip())

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def cosine_similarity(a, b):
    return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

class Embedding:
    def __init__(self, word_list, vector_list):
        self.word2id = {}
        self.id2word = {}
        self.word2id['[UNK]'] = 0
        self.id2word[0] = '[UNK]'
        for i, w in enumerate(word_list):
            self.word2id[w] = i + 1
            self.id2word[i+1] = w
        avg_vector = np.array(vector_list[0])
        for i in range(1, len(vector_list)):
            avg_vector += np.array(vector_list[i])
        avg_vector /= len(vector_list)

        self.vector_list = [avg_vector] + vector_list

    def __getitem__(self, word):
        if word not in self.word2id:
            return self.vector_list[0]
        else:
            return self.vector_list[self.word2id[word]]


class DenseEmbedder:
    def __init__(self, embedding, tokenizer, stop_list=[]):
        self.embedding = embedding
        self.stop_list = set(stop_list)
        self.tokenizer = tokenizer

    def remove_stop_words(self, sent):
        tokenized_sent = self.tokenizer(sent, cut_all=True)
        ret = []
        for w in tokenized_sent:
            if w in self.stop_list:
                continue
            ret.append(w)
        return sorted(list(set(ret)))

    def encode(self, sent, mode='average'):
        tokenized_sent_without_stopwords = self.remove_stop_words(sent)
        if len(tokenized_sent_without_stopwords) == 0:
            return self.embedding['[UNK]']
        encoded_words = []
        for w in tokenized_sent_without_stopwords:
            encoded_words.append(self.embedding[w])
        if mode == 'average':
            return np.mean(encoded_words, 0)

    def get_embedding_vectors(self, sent):
        tokenized_sent_without_stopwords = self.remove_stop_words(sent)
        encoded_words = []
        for w in tokenized_sent_without_stopwords:
            encoded_words.append(self.embedding[w])
        return encoded_words

def search_evidence(queries, context_sents, dense_encoded_sents, dense_embedder, bert_embedder, beam_size=2):
    retrieved_evidence = []
    for query in queries:
        candidates = []
        # first retrieval
        query_tokens = dense_embedder.remove_stop_words(query)
        first_step_dense_encoded_query =  dense_embedder.encode(query)
        first_step_results = []
        for i, dense_encoded_sent in enumerate(dense_encoded_sents):
            sim = cosine_similarity(first_step_dense_encoded_query, dense_encoded_sent)
            first_step_results.append([context_sents[i], sim])
        first_step_results = sorted(first_step_results, key=lambda x: x[1], reverse=True)
        # add to candidates
        for i in range(beam_size):
            candidates.append((first_step_results[i][0], 1))

        # second retrieval, beam search
        for first_step_evidence, first_step_score in first_step_results[:beam_size]:
            second_step_results = []
            query_embedding_vectors = dense_embedder.get_embedding_vectors(query)
            first_step_evidence_embedding_vectors = dense_embedder.get_embedding_vectors(first_step_evidence)
            if len(first_step_evidence_embedding_vectors) == 0:
                continue
            attention_matrix = np.matmul(query_embedding_vectors, np.array(first_step_evidence_embedding_vectors).T)
            # attention_matrix = [query len, sent len]
            max_query_aware_attention = softmax(- (np.max(attention_matrix, 1)))
            second_step_dense_encoded_query = np.dot(max_query_aware_attention, query_embedding_vectors)
            for j, dense_encoded_sent in enumerate(dense_encoded_sents):
                sim = cosine_similarity(second_step_dense_encoded_query, dense_encoded_sent)
                second_step_results.append([context_sents[j], sim])
            second_step_results = sorted(second_step_results, key=lambda x: x[1], reverse=True)

            # add composed evidence to candidates
            for j in range(beam_size):
                sent1 = first_step_evidence
                sent2 = second_step_results[j][0]
                index1 = context_sents.index(sent1)
                index2 = context_sents.index(sent2)
                composed_evidence = None
                if index1 == index2:
                    continue
                elif index1 < index2:
                    composed_evidence = sent1 + sent2
                else:
                    composed_evidence = sent2 + sent1
                candidates.append((composed_evidence, 2))

        # final ranker
        final_results = []
        bert_encoded_query = bert_embedder.encode(query, convert_to_tensor=True)
        for cand, step_cnt in candidates:
            bert_encoded_sent = bert_embedder.encode(cand, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(bert_encoded_query, bert_encoded_sent)
            final_results.append((cand, float(sim[0][0].item()), step_cnt))
        final_results = sorted(final_results, key=lambda x: x[1], reverse=True)

        retrieved_evidence.append([{'text': text, 'score': score, 'step_cnt': step_cnt} for text, score, step_cnt in final_results])
    return retrieved_evidence

def iteratively_retrieve_evidence(original_data_path, save_data_path):
    dataset = json_load(original_data_path)
    new_dataset = {
        'version': dataset['version'] + ' with iteratively retrieved evidence', 
        'data': []
    }
    for item in tqdm(dataset['data']):
        context_sents = sent_tokenize(item['context'])
        dense_encoded_sents = [dense_embedder.encode(s) for s in context_sents]
        for j, qas in enumerate(item['qas']):
            iterative_retrieved_evidence = search_evidence(qas['options'], context_sents, dense_encoded_sents, dense_embedder, bert_embedder, beam_size=2)
            item['qas'][j]['iteratively_retrieved_evidence'] = iterative_retrieved_evidence
        new_dataset['data'].append(item)
    json_dump(new_dataset, save_data_path)

   
if __name__ == '__main__':
    
    '''
    Loading external resources:
    1. Word vectors
    sgns.merge.char_word-list and sgns.merge.char_vector-list 
    are word vectors extracted from [1], 
    containing all the words appearing in VGaokao.
    [1] https://github.com/Embedding/Chinese-Word-Vectors

    2. Sentence-BERT
    We use the distiluse-base-multilingual-cased-v1 model in [2].
    [2] https://www.sbert.net/docs/pretrained_models.html#multi-lingual-models
    '''

    word_list = pk.load(open("../data/external/word_vectors/sgns.merge.char_word-list", 'rb'))
    print(len(word_list))
    vector_list = pk.load(open("../data/external/word_vectors/sgns.merge.char_vector-list", 'rb'))
    print("vector loaded")
    embedding = Embedding(word_list, vector_list)
    dense_embedder = DenseEmbedder(embedding, jieba.cut, simple_stop_list)
    if os.path.exists('../data/external/distiluse-base-multilingual-cased-v1'):
        bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')
    else:
        bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')
    # If you cannot download the model with SentenceTransformer,
    # you can download the model from
    # https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/
    # , and load it from a local folder by
    # bert_embedder = SentenceTransformer('../data/external/distiluse-base-multilingual-cased-v1')

    '''
    Start extracting evidence
    '''
    iteratively_retrieve_evidence('../data/raw/train.json', '../data/processed/train_mrc_iterative.json')
    iteratively_retrieve_evidence('../data/raw/test.json', '../data/processed/test_mrc_iterative.json')

