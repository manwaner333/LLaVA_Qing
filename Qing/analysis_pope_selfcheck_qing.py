import pickle
# import numpy as np
# from scipy.stats import entropy
from tqdm import tqdm
import os
# import json
from sklearn.metrics import precision_recall_curve, auc
# from llava.model.builder import load_pretrained_model
# import matplotlib.pyplot as plt
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# import random
# import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import torch
import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram, SelfCheckNLI
nlp = spacy.load("en_core_web_sm")



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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_mqag = SelfCheckMQAG(device=device)  # set device to 'cuda' if GPU is available
    selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
    selfcheck_ngram = SelfCheckNgram(n=1)

    human_label_detect_False = {}
    human_label_detect_True = {}

    idxs = []
    with open("result/coco2014_val/answer_coco_pope_adversarial_new_prompt.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            idx = line[1].split('.')[0]
            if idx not in idxs:
                human_label_detect_True[idx] = []
                human_label_detect_False[idx] = []
                idxs.append(idx)
            if line[6] == "accurate":
                true_score = 1.0
                false_score = 0.0
            else:
                true_score = 0.0
                false_score = 1.0
            human_label_detect_True[idx].append(true_score)
            human_label_detect_False[idx].append(false_score)

    # for idx in idxs:
    path = f"result/coco2014_val/answer_coco_pope_adversarial_new_prompt.bin"
    with open(path, "rb") as f:
        responses = pickle.load(f)

    path1 = f"result/coco2014_val/answer_coco_pope_adversarial_new_prompt_random_1.bin"
    with open(path1, "rb") as f1:
        responses1 = pickle.load(f1)

    path2 = f"result/coco2014_val/answer_coco_pope_adversarial_new_prompt_random_2.bin"
    with open(path2, "rb") as f2:
        responses2 = pickle.load(f2)

    path3 = f"result/coco2014_val/answer_coco_pope_adversarial_new_prompt_random_3.bin"
    with open(path3, "rb") as f3:
        responses3 = pickle.load(f3)

    mqag_scores = {}
    bert_scores = {}
    ngram_scores = {}
    nli_scores = {}

    for idx, response in tqdm(responses.items()):
        # if index < 231:
        #     index += 1
        #     continue
        # print(idx)
        # from the cache
        passage = response['text']
        sentences = response['sentences']
        num_sentences = len(sentences)

        sample1 = responses1[idx]['text']
        sample2 = responses2[idx]['text']
        sample3 = responses3[idx]['text']

        sent_scores_mqag = selfcheck_mqag.predict(
                sentences=sentences,               # list of sentences
                passage=passage,                   # passage (before sentence-split)
                sampled_passages=[sample1, sample2, sample3], # list of sampled passages
                num_questions_per_sent=5,          # number of questions to be drawn
                scoring_method='bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
                beta1=0.8, beta2=0.8,            # additional params depending on scoring_method
            )
        mqag_scores[idx] = sent_scores_mqag

        sent_scores_bertscore = selfcheck_bertscore.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=[sample1, sample2, sample3],  # list of sampled passages
        )
        bert_scores[idx] = sent_scores_bertscore


        sent_scores_ngram = selfcheck_ngram.predict(
            sentences=sentences,
            passage=passage,
            sampled_passages=[sample1, sample2, sample3],
        )
        ngram_scores[idx] = sent_scores_ngram['sent_level']['avg_neg_logprob']


        selfcheck_nli = SelfCheckNLI(device=device)  # set device to 'cuda' if GPU is available
        sent_scores_nli = selfcheck_nli.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=[sample1, sample2, sample3],  # list of sampled passages
        )
        nli_scores[idx] = sent_scores_nli

    mqag_scores_path = "result/coco2014_val/answer_pope_selfcheck_mqag.bin"
    bert_scores_path = "result/coco2014_val/answer_pope_selfcheck_bert.bin"
    ngram_scores_path = "result/coco2014_val/answer_pope_selfcheck_ngram.bin"
    nli_scores_path = "result/coco2014_val/answer_pope_selfcheck_nli.bin"

    mqag_scores_file = os.path.expanduser(mqag_scores_path)
    os.makedirs(os.path.dirname(mqag_scores_file), exist_ok=True)

    bert_scores_file = os.path.expanduser(bert_scores_path)
    os.makedirs(os.path.dirname(bert_scores_file), exist_ok=True)

    ngram_scores_file = os.path.expanduser(ngram_scores_path)
    os.makedirs(os.path.dirname(ngram_scores_file), exist_ok=True)

    nli_scores_file = os.path.expanduser(nli_scores_path)
    os.makedirs(os.path.dirname(nli_scores_file), exist_ok=True)


    with open(mqag_scores_file, 'wb') as file:
        pickle.dump(mqag_scores, file)

    with open(bert_scores_file, 'wb') as file:
        pickle.dump(bert_scores, file)

    with open(ngram_scores_file, 'wb') as file:
        pickle.dump(ngram_scores, file)

    with open(nli_scores_file, 'wb') as file:
        pickle.dump(nli_scores, file)


    # Pb1, Rb1 = get_PR_with_human_labels(mqag_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    # Pb2, Rb2 = get_PR_with_human_labels(bert_scores, human_label_detect_False, pos_label=1)
    # Pb3, Rb3 = get_PR_with_human_labels(ngram_scores, human_label_detect_False, pos_label=1, oneminus_pred=True)
    # Pb4, Rb4 = get_PR_with_human_labels(nli_scores, human_label_detect_False, pos_label=1)

    # Pb1, Rb1 = get_PR_with_human_labels(mqag_scores, human_label_detect_True, pos_label=1, oneminus_pred=False)
    # Pb2, Rb2 = get_PR_with_human_labels(bert_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
    # Pb3, Rb3 = get_PR_with_human_labels(ngram_scores, human_label_detect_True, pos_label=1, oneminus_pred=False)
    # Pb4, Rb4 = get_PR_with_human_labels(nli_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)

    # print("-----------------------")
    # print("Baseline1: Avg(logP)")
    # print_AUC(Pb1, Rb1)
    # print("-----------------------")
    # print("Baseline2: Avg(H)")
    # print_AUC(Pb2, Rb2)
    # print("-----------------------")
    # print("Baseline3: Max(-logP)")
    # print_AUC(Pb3, Rb3)
    # print("-----------------------")
    # print("Baseline4: Max(H)")
    # print_AUC(Pb4, Rb4)
    #
    # arr = []
    # for v in human_label_detect_False.values():
    #     arr.extend(v)
    # random_baseline = np.mean(arr)
    #
    # # with human label, Detecting Non-factual*
    # plt.figure(figsize=(5.5, 4.5))
    # plt.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    # plt.plot(Rb1, Pb1, '-', label='Avg(logP)')
    # plt.plot(Rb2, Pb2, '-', label='Avg(H)')
    # plt.plot(Rb3, Pb3, '-', label='Max(-logP)')
    # plt.plot(Rb4, Pb4, '-', label='Max(H)')
    # plt.legend()
    # plt.ylabel("Precision")
    # plt.xlabel("Recall")
    # plt.show()

















