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
    human_label_detect_False_h = {}
    human_label_detect_True = {}

    idxs = []
    with open("../playground/data/gqa/llava_gqa_testdev_balanced_filter_prompt.json", "r") as f:
        for line in f:
            ele = json.loads(line)
            idx = ele['question_id']
            idxs.append(idx)
            # human_label_detect_False[idx] = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            # human_label_detect_True[idx] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]


    # for idx in idxs:
    path = f"../result/gqa/answer_gqa_testdev_balanced_filter.bin"
    with open(path, "rb") as f:
        responses = pickle.load(f)

    path = f"../result/gqa/answer_gqa_testdev_balanced_filter_radom_1.bin"
    with open(path, "rb") as f:
        responses1 = pickle.load(f)

    baseline1_scores = {}  # average_logprob
    baseline2_scores = {}  # lowest_logprob
    baseline3_scores = {}  # average_entropy5
    baseline4_scores = {}  # highest_entropy5

    for idx, response in responses.items():
        # from the cache
        passage = response['text']
        tokens = response['logprobs']['tokens']
        token_logprobs = response['logprobs']['token_logprobs']
        top_logprobs = response['logprobs']['top_logprobs']
        sentences = response['sentences']
        num_sentences = len(sentences)
        # human_label_detect_False[idx] = [random.choice([0, 1]) for _ in range(num_sentences)]
        # human_label_detect_True[idx] = [1 - ele for ele in human_label_detect_False[idx]]

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

        baseline1_scores[idx] = average_logprob_sent_level
        baseline2_scores[idx] = average_entropy5_sent_level
        baseline3_scores[idx] = lowest_logprob_sent_level
        baseline4_scores[idx] = highest_entropy5_sent_level

    Pb1, Rb1 = get_PR_with_human_labels(baseline1_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    Pb1, Rb1 = get_PR_with_human_labels(baseline1_scores, human_label_detect_True, pos_label=1, oneminus_pred=False)
    Pb2, Rb2 = get_PR_with_human_labels(baseline2_scores, human_label_detect_False, pos_label=1)
    Pb3, Rb3 = get_PR_with_human_labels(baseline3_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    Pb4, Rb4 = get_PR_with_human_labels(baseline4_scores, human_label_detect_False, pos_label=1)
    print("-----------------------")
    print("Baseline1: Avg(logP)")
    print_AUC(Pb1, Rb1)
    print("-----------------------")
    print("Baseline2: Avg(H)")
    print_AUC(Pb2, Rb2)
    print("-----------------------")




