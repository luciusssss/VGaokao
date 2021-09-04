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
from pprint import pprint

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

###############################################################

def to_nli(dataset, mode='iterative'):
    new_dataset = {
        'version': dataset['version'],
        'data': []
    }
    for item in tqdm(dataset['data']):
        for qas in item['qas']:
            for i in range(4):
                if mode == 'iterative':
                    new_dataset['data'].append({
                        "sentence1": qas['iteratively_retrieved_evidence'][i][0]['text'] if len(qas['iteratively_retrieved_evidence'][i]) > 0 else "无",
                        "sentence2": qas['options'][i],
                        "label": qas['correctness'][i],
                        "qid": qas['qid'],
                        "opid": i,
                        "neighbor_type": 'iterative'
                    })
    return new_dataset

def convert_to_hinge_pair(dataset):
    new_dataset = {
        'verson': dataset['version'] + ' hinge pair',
        'data': []
    }
    for i in range(int(len(dataset['data'])/4)):
        pos_cnt = 0
        for j in range(4):
            if dataset['data'][4 * i + j]['label'] == 1:
                pos_cnt += 1
        if pos_cnt == 1:
            pos_id = None
            for j in range(4):
                if dataset['data'][4 * i + j]['label'] == 1:
                    pos_id = 4 * i + j
                    break
            for j in range(4):
                if 4 * i + j == pos_id:
                    continue
                new_dataset['data'].append({
                    'sentence1_pos': dataset['data'][pos_id]['sentence1'],
                    'sentence2_pos': dataset['data'][pos_id]['sentence2'],
                    'sentence1_neg': dataset['data'][4 * i + j]['sentence1'],
                    'sentence2_neg': dataset['data'][4 * i + j]['sentence2'],
                    'label': 1
                })
        elif pos_cnt == 3:
            neg_id = None
            for j in range(4):
                if dataset['data'][4 * i + j]['label'] == 0:
                    neg_id = 4 * i + j
                    break
            for j in range(4):
                if 4 * i + j == neg_id:
                    continue
                new_dataset['data'].append({
                    'sentence1_pos': dataset['data'][4 * i + j]['sentence1'],
                    'sentence2_pos': dataset['data'][4 * i + j]['sentence2'],
                    'sentence1_neg': dataset['data'][neg_id]['sentence1'],
                    'sentence2_neg': dataset['data'][neg_id]['sentence2'],
                    'label': 1
                })
        else:
            print('Wrong case', i)
    return new_dataset


if __name__ == '__main__':
    train_data = json_load('../data/processed/train_mrc_iterative.json')
    test_data = json_load('../data/processed/test_mrc_iterative.json')
    new_train_data = to_nli(train_data, mode='iterative')
    new_test_data = to_nli(test_data, mode='iterative')
    json_dump(new_train_data, '../data/processed/train_nli_iterative.json')
    json_dump(new_test_data, '../data/processed/test_nli_iterative.json')

    new_train_data = convert_to_hinge_pair(new_train_data)
    new_test_data = convert_to_hinge_pair(new_test_data)
    json_dump(new_train_data, '../data/processed/train_nli_iterative-hinge.json')
    json_dump(new_test_data, '../data/processed/test_nli_iterative-hinge.json')


    