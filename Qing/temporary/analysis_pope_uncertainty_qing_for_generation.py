import pickle
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from datasets import load_dataset
import os
import json
from sklearn.metrics import precision_recall_curve, auc
from llava.model.builder import load_pretrained_model
import matplotlib.pyplot as plt
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import random
import pandas as pd
import csv

def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled

def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)
    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]
    unroll_labels = unroll_pred(human_labels, indices)
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    return P, R

def print_AUC(P, R):
    print("AUC: {:.2f}".format(auc(R, P)*100))



if __name__ == "__main__":
    human_label_detect_False = {}
    human_label_detect_True = {}

    idxs = []
    stop_row = 409
    with open("../result/coco2014_val/answer_coco_pope_adversarial_new_prompt.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row_number, line in enumerate(reader, start=1):
            if row_number > stop_row:
                break
            idx = line[1].split('.')[0]
            if idx not in idxs:
                human_label_detect_True[idx] = []
                human_label_detect_False[idx] = []
                idxs.append(idx)
            if line[7] == "accurate":
                true_score = 1.0
                false_score = 0.0
            elif line[7] == "inaccurate":
                true_score = 0.0
                false_score = 1.0
            human_label_detect_True[idx].append(true_score)
            human_label_detect_False[idx].append(false_score)

    # analysis true ratio:
    true_values = []
    for key, value in human_label_detect_True.items():
        true_values.extend(value)
    total_num = len(true_values)
    print("The total number of sentences is: {}; The ratio of true values is: {}".format(total_num, true_values.count(1.0)/total_num))

    # for idx in idxs:
    path = f"../result/coco2014_val/answer_coco_pope_adversarial_new_prompt_add_states.bin"
    with open(path, "rb") as f:
        responses = pickle.load(f)

    # 读取selfcheck的数据
    # bert
    path_bert = f"../result/coco2014_val/answer_pope_selfcheck_bert.bin"
    with open(path_bert, "rb") as f:
        values_bert = pickle.load(f)

    # mqag
    path_mqag = f"../result/coco2014_val/answer_pope_selfcheck_mqag.bin"
    with open(path_mqag, "rb") as f:
        values_mqag = pickle.load(f)

    # ngram
    path_ngram = f"../result/coco2014_val/answer_pope_selfcheck_ngram.bin"
    with open(path_ngram, "rb") as f:
        values_ngram = pickle.load(f)

    # nli
    path_nli = f"../result/coco2014_val/answer_pope_selfcheck_nli.bin"
    with open(path_nli, "rb") as f:
        values_nli = pickle.load(f)

    average_logprob_scores = {}  # average_logprob
    average_entropy5_scores = {}  # lowest_logprob
    lowest_logprob_scores = {}  # average_entropy5
    highest_entropy5_scores = {}  # highest_entropy5
    bert_scores = {}
    mqag_scores = {}
    ngram_scores = {}
    nli_scores = {}

    for idx, response in responses.items():
        # from the cache
        if idx not in idxs:
            break
        passage = response['text']
        tokens = response['logprobs']['tokens']
        token_logprobs = response['logprobs']['token_logprobs']
        top_logprobs = response['logprobs']['top_logprobs']
        sentences = response['sentences']
        num_sentences = len(sentences)

        if num_sentences != len(human_label_detect_True[idx]):
            print("There are some information about the annocatated sentences.")

        average_logprob_sent_level = [None for _ in range(num_sentences)]
        lowest_logprob_sent_level = [None for _ in range(num_sentences)]
        average_entropy5_sent_level = [None for _ in range(num_sentences)]
        highest_entropy5_sent_level = [None for _ in range(num_sentences)]


        for sent_i, sentence in enumerate(sentences):
            # sentence exist in the passage, so we need to find where it is [i1, i2]
            sentence_tf = "".join(sentence.split(" "))
            xarr = [i for i in range(len(tokens))]
            for i1 in xarr:
                mystring = "".join(tokens[i1:])
                if sentence_tf not in mystring:
                    break
            i1 = i1 - 1
            for i2 in xarr[::-1]:
                mystring = "".join(tokens[i1:i2 + 1])
                if sentence_tf not in mystring:
                    break
            i2 = i2 + 1
            mystring = "".join(tokens[i1:i2 + 1])
            average_logprob = np.mean(token_logprobs[i1:i2 + 1])
            lowest_logprob = np.min(token_logprobs[i1:i2 + 1])
            entropy5s = []
            for top5_tokens in top_logprobs[i1:i2 + 1]:
                logprob_of_top5_tokens = [x[1] for x in list(top5_tokens.items())]
                logprob_of_top5_tokens = np.array(logprob_of_top5_tokens)
                prob_of_top5_tokens = np.exp(logprob_of_top5_tokens)
                total_prob_of_top5 = prob_of_top5_tokens.sum()
                normalized_prob = prob_of_top5_tokens / total_prob_of_top5
                # this was designed to be PPL, and intial results showed a better performance level than just entropy
                entropy5 = 2 ** (entropy(normalized_prob, base=2))
                entropy5s.append(entropy5)
            average_entropy5 = np.mean(entropy5s)
            highest_entropy5 = np.max(entropy5s)

            average_logprob_sent_level[sent_i] = average_logprob
            lowest_logprob_sent_level[sent_i] = lowest_logprob
            average_entropy5_sent_level[sent_i] = average_entropy5
            highest_entropy5_sent_level[sent_i] = highest_entropy5

        average_logprob_scores[idx] = average_logprob_sent_level
        average_entropy5_scores[idx] = average_entropy5_sent_level
        lowest_logprob_scores[idx] = lowest_logprob_sent_level
        highest_entropy5_scores[idx] = highest_entropy5_sent_level
        bert_scores[idx] = values_bert[idx]
        mqag_scores[idx] = values_mqag[idx]
        ngram_scores[idx] = values_ngram[idx]
        nli_scores[idx] = values_nli[idx]

    # Pb1, Rb1 = get_PR_with_human_labels(baseline1_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    # Pb2, Rb2 = get_PR_with_human_labels(baseline2_scores, human_label_detect_False, pos_label=1)
    # Pb3, Rb3 = get_PR_with_human_labels(baseline3_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    # Pb4, Rb4 = get_PR_with_human_labels(baseline4_scores, human_label_detect_False, pos_label=1)

    # True
    # uncertainty
    Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores, human_label_detect_True, pos_label=1, oneminus_pred=False)
    Pb_average_entropy5, Rb_average_entropy5 = get_PR_with_human_labels(average_entropy5_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
    Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label_detect_True, pos_label=1, oneminus_pred=False)
    Pb_highest_entropy5, Rb_highest_entropy5 = get_PR_with_human_labels(highest_entropy5_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)

    # selfcheck
    Pb_bert, Rb_bert = get_PR_with_human_labels(bert_scores, human_label_detect_True, pos_label=1,  oneminus_pred=True)
    Pb_mqag, Rb_mqag = get_PR_with_human_labels(mqag_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
    Pb_ngram, Rb_ngram = get_PR_with_human_labels(ngram_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
    Pb_nli, Rb_nli = get_PR_with_human_labels(nli_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)

    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb_average_logprob, Rb_average_logprob)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb_average_entropy5, Rb_average_entropy5)
    print("-----------------------")
    print("Baseline3: Max(-logP)")
    print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
    print("-----------------------")
    print("Baseline4: Max(H)")
    print_AUC(Pb_highest_entropy5, Rb_highest_entropy5)
    print("-----------------------")
    print("Baseline5: Bert")
    print_AUC(Pb_bert, Rb_bert)
    print("-----------------------")
    print("Baseline6: Mqag")
    print_AUC(Pb_mqag, Rb_mqag)
    print("-----------------------")
    print("Baseline7: Ngram")
    print_AUC(Pb_ngram, Rb_ngram)
    print("-----------------------")
    print("Baseline8: Nli")
    print_AUC(Pb_nli, Rb_nli)

    arr = []
    for v in human_label_detect_True.values():
        arr.extend(v)
    random_baseline = np.mean(arr)

    # with human label, Detecting Non-factual*
    plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label='Avg(logP)')
    plt.plot(Rb_average_entropy5, Pb_average_entropy5, '-', label='Avg(H)')
    plt.plot(Rb_lowest_logprob, Pb_lowest_logprob,  '-', label='Max(-logP)')
    plt.plot(Rb_highest_entropy5, Pb_highest_entropy5, '-', label='Max(H)')
    plt.plot(Rb_bert, Pb_bert, '-', label='Bert')
    plt.plot(Rb_mqag, Pb_mqag, '-', label='Mqag')
    plt.plot(Rb_ngram, Pb_ngram, '-', label='Ngram')
    plt.plot(Rb_nli, Pb_nli, '-', label='Nli')
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()

    # False
    # uncertainty
    Pb_average_logprob, Rb_average_logprob = get_PR_with_human_labels(average_logprob_scores, human_label_detect_False,
                                                                      pos_label=1, oneminus_pred=True)
    Pb_average_entropy5, Rb_average_entropy5 = get_PR_with_human_labels(average_entropy5_scores,
                                                                        human_label_detect_False, pos_label=1,
                                                                        oneminus_pred=False)
    Pb_lowest_logprob, Rb_lowest_logprob = get_PR_with_human_labels(lowest_logprob_scores, human_label_detect_False,
                                                                    pos_label=1, oneminus_pred=True)
    Pb_highest_entropy5, Rb_highest_entropy5 = get_PR_with_human_labels(highest_entropy5_scores,
                                                                        human_label_detect_False, pos_label=1,
                                                                        oneminus_pred=False)

    # selfcheck
    Pb_bert, Rb_bert = get_PR_with_human_labels(bert_scores, human_label_detect_False, pos_label=1, oneminus_pred=False)
    Pb_mqag, Rb_mqag = get_PR_with_human_labels(mqag_scores, human_label_detect_False, pos_label=1, oneminus_pred=False)
    Pb_ngram, Rb_ngram = get_PR_with_human_labels(ngram_scores, human_label_detect_False, pos_label=1,
                                                  oneminus_pred=False)
    Pb_nli, Rb_nli = get_PR_with_human_labels(nli_scores, human_label_detect_False, pos_label=1, oneminus_pred=False)

    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb_average_logprob, Rb_average_logprob)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb_average_entropy5, Rb_average_entropy5)
    print("-----------------------")
    print("Baseline3: Max(-logP)")
    print_AUC(Pb_lowest_logprob, Rb_lowest_logprob)
    print("-----------------------")
    print("Baseline4: Max(H)")
    print_AUC(Pb_highest_entropy5, Rb_highest_entropy5)
    print("-----------------------")
    print("Baseline5: Bert")
    print_AUC(Pb_bert, Rb_bert)
    print("-----------------------")
    print("Baseline6: Mqag")
    print_AUC(Pb_mqag, Rb_mqag)
    print("-----------------------")
    print("Baseline7: Ngram")
    print_AUC(Pb_ngram, Rb_ngram)
    print("-----------------------")
    print("Baseline8: Nli")
    print_AUC(Pb_nli, Rb_nli)

    arr = []
    for v in human_label_detect_False.values():
        arr.extend(v)
    random_baseline = np.mean(arr)

    # with human label, Detecting Non-factual*
    plt.figure(figsize=(5.5, 4.5))
    plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    plt.plot(Rb_average_logprob, Pb_average_logprob, '-', label='Avg(logP)')
    plt.plot(Rb_average_entropy5, Pb_average_entropy5, '-', label='Avg(H)')
    plt.plot(Rb_lowest_logprob, Pb_lowest_logprob, '-', label='Max(-logP)')
    plt.plot(Rb_highest_entropy5, Pb_highest_entropy5, '-', label='Max(H)')
    plt.plot(Rb_bert, Pb_bert, '-', label='Bert')
    plt.plot(Rb_mqag, Pb_mqag, '-', label='Mqag')
    plt.plot(Rb_ngram, Pb_ngram, '-', label='Ngram')
    plt.plot(Rb_nli, Pb_nli, '-', label='Nli')
    plt.legend()
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()






