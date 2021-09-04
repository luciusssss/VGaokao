import json
import codecs
import random
import sys
import math
import numpy as np

ave = lambda x : sum(x)/len(x)
codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
json_dumpsl = lambda d: json.dumps(d, ensure_ascii=False)

#######################################

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("参数个数错误")
        exit(1)
    mrc_dataset = json_load(sys.argv[1])
    nli_prob_file = sys.argv[2]
    output_file = sys.argv[3]

    nli_prob = []
    with open(nli_prob_file, 'r') as f:
        for line in f.readlines():
            line = line.split()
            p0 = float(line[1])
            p1 = float(line[2])
            exp0 = math.exp(p0)
            exp1 = math.exp(p1)
            nli_prob.append(exp1/(exp0+exp1))

    qas_cnt = 0
    correct_cnt = 0

    correct_option_cnt = 0
    option_cnt = 0

    output = {
        'results': []
    }
    for item in mrc_dataset['data']:
        for qas in item['qas']:
            cur_prob = nli_prob[4*qas_cnt:4*qas_cnt+4]
            for i in range(4):
                if (cur_prob[i] > 0.5 and qas['correctness'][i] == 1) or (cur_prob[i] < 0.5 and qas['correctness'][i] == 0):
                    correct_option_cnt += 1


            qas_cnt += 1
            prediction = None
            if "不" in qas['question'] or "错误" in qas['question'] or "有误" in qas['question']:
                prediction = np.argmin(cur_prob)
            else:
                prediction = np.argmax(cur_prob)
            if prediction == ord(qas['answer']) - ord('A'):
                correct_cnt += 1
            output['results'].append({
                'qid': qas['qid'],
                'prob': cur_prob,
                'question': qas['question'],
                'prediction': int(prediction),
                'answer': int(ord(qas['answer']) - ord('A'))
            })
    output['accuracy'] = correct_cnt/qas_cnt
    print("Option-level Acc.", correct_option_cnt/(qas_cnt*4))
    print("Question-level Acc:", correct_cnt/qas_cnt)
    json_dump(output, output_file)