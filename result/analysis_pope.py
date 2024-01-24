import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import tqdm
import pandas as pd


# read json file

def calculate_metrics(file_name, average):
    data_pred = []
    data_true = []
    yes_pred = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            # no(No): 0; yes(Yes): 1:
            if dic['label'] == 'yes':
                data_true.append(1)
            else:
                data_true.append(0)
                if dic['text'] == 'Yes':
                    yes_pred.append(1)

            if dic['text'] == 'Yes':
                data_pred.append(1)
            elif dic['text'] == 'No':
                data_pred.append(0)
    acc = accuracy_score(data_true, data_pred)
    precision = precision_score(data_true, data_pred, average=average)
    recall = recall_score(data_true, data_pred, average=average)
    f1 = f1_score(data_true, data_pred, average=average)
    # yes_ratio = data_pred.count(1)/len(data_pred)
    yes_ratio = yes_pred.count(1) / (3000 - data_true.count(1))

    return acc, precision, recall, f1, yes_ratio

def calculate_metrics(file_ad):
    ans_file = file_ad
    label_file = file_ad

    answers = [json.loads(q) for q in open(ans_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['answer'] = 'no'
        else:
            answer['answer'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['answer'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


def analysis_hidden_states(file_name, figure_name, layer):
    # Generate a synthetic high-dimensional dataset
    hidden_states = []
    labels = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            dic = json.loads(line)
            hidden_states.append(dic[layer])
            # (1) text == label, then 1; else: 0
            if dic['text'].lower() == dic['label'].lower():
                labels.append(1)
            else:
                labels.append(0)

    # Applying PCA to reduce dimensions to 2
    pca = PCA(n_components=3)
    X_pca_3d = pca.fit_transform(hidden_states)

    # Applying K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(X_pca_3d)

    # Plotting in 3D
    fig = plt.figure(figsize=(14, 8))

    # Plotting the PCA-reduced data colored by true labels
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(X_pca_3d)):
        if labels[i] == 1:
            s1 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
        else:
            s2 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
    ax1.set_title('True Labels in 3D PCA-reduced Space')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend((s1, s2), ("True", "False"), loc='best')

    # Plotting the PCA-reduced data colored by cluster labels
    ax2 = fig.add_subplot(122, projection='3d')
    for i in range(len(X_pca_3d)):
        if clusters[i] == 1:
            o1 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
        else:
            o2 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
    ax2.set_title('K-Means Clusters in 3D PCA-reduced Space')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend((o1, o2), ("Class 1", "Class 2"), loc='best')
    plt.tight_layout()
    path = 'coco2014_val/' + figure_name + '_' + layer + '.png'
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, format='png')
    plt.show()

def calculate_consistency_between_respones(file_1, file_2, file_3, file_4, file_5, file_6, file_7, file_8, file_9):
    labels = []
    prompts = []
    question_ids = []
    texts_1 = []
    texts_2 = []
    texts_3 = []
    texts_4 = []
    texts_5 = []
    texts_6 = []
    texts_7 = []
    texts_8 = []
    texts_9 = []

    # file_ad_1
    with open(file_1, "r") as f1:
        for line in f1.readlines():
            dic = json.loads(line)
            texts_1.append(dic['text'])
            labels.append(dic['label'])
            prompts.append(dic['prompt'])
            question_ids.append(dic['question_id'])

    # file_ad_2
    with open(file_2, "r") as f2:
        for line in f2.readlines():
            dic = json.loads(line)
            texts_2.append(dic['text'])

    # file_ad_3
    with open(file_3, "r") as f3:
        for line in f3.readlines():
            dic = json.loads(line)
            texts_3.append(dic['text'])

    # file_ad_4
    with open(file_4, "r") as f4:
        for line in f4.readlines():
            dic = json.loads(line)
            texts_4.append(dic['text'])

    # file_ad_5
    with open(file_5, "r") as f5:
        for line in f5.readlines():
            dic = json.loads(line)
            texts_5.append(dic['text'])

    # file_ad_6
    with open(file_6, "r") as f6:
        for line in f6.readlines():
            dic = json.loads(line)
            texts_6.append(dic['text'])

    # file_ad_7
    with open(file_7, "r") as f7:
        for line in f7.readlines():
            dic = json.loads(line)
            texts_7.append(dic['text'])

    # file_ad_8
    with open(file_8, "r") as f8:
        for line in f8.readlines():
            dic = json.loads(line)
            texts_8.append(dic['text'])

    # file_ad_9
    with open(file_9, "r") as f9:
        for line in f9.readlines():
            dic = json.loads(line)
            texts_9.append(dic['text'])

    output_excel = {'question_ids': [], 'prompts': [], 'labels': [], 'answer_1': [], 'answer_2': [], 'answer_3': [],
                    'answer_4': [], 'answer_5': [], 'answer_6': [], 'answer_7': [], 'answer_8': [], 'answer_9': []}

    output_excel['question_ids'] = question_ids
    output_excel['prompts'] = prompts
    output_excel['labels'] = labels
    output_excel['answer_1'] = texts_1
    output_excel['answer_2'] = texts_2
    output_excel['answer_3'] = texts_3
    output_excel['answer_4'] = texts_4
    output_excel['answer_5'] = texts_5
    output_excel['answer_6'] = texts_6
    output_excel['answer_7'] = texts_7
    output_excel['answer_8'] = texts_8
    output_excel['answer_9'] = texts_9
    res = []
    for i in range(0, 3000):
        if not (output_excel['answer_1'][i] == output_excel['answer_2'][i] == output_excel['answer_3'][i] ==
                output_excel['answer_4'][i] == output_excel['answer_5'][i]
                == output_excel['answer_6'][i] == output_excel['answer_7'][i] == output_excel['answer_8'][i] ==
                output_excel['answer_9'][i]):
            res.append([output_excel['question_ids'][i], output_excel['prompts'][i], output_excel['labels'][i],
                        output_excel['answer_1'][i],
                        output_excel['answer_2'][i], output_excel['answer_3'][i], output_excel['answer_4'][i],
                        output_excel['answer_5'][i],
                        output_excel['answer_6'][i], output_excel['answer_7'][i], output_excel['answer_8'][i],
                        output_excel['answer_9'][i]])
    return len(res)/len(question_ids)


if __name__ == "__main__":

    ##### calculate metric ######
    # file_ad = "coco2014_val/answer_coco_pope_adversarial.json"
    # file_po = "coco2014_val/answer_coco_pope_popular.json"
    # file_ra = "coco2014_val/answer_coco_pope_random.json"
    # get the metric of the results: accuracy, precision, recall, f1 and yes_ratio
    # average = "binary"
    # acc_ad, precision_ad, recall_ad, f1_ad, yes_ratio_ad = calculate_metrics(file_ad, average)
    # acc_po, precision_po, recall_po, f1_po, yes_ratio_po = calculate_metrics(file_po, average)
    # acc_ra, precision_ra, recall_ra, f1_ra, yes_ratio_ra = calculate_metrics(file_ra, average)
    #
    # print("The result of random analysis is: accuracy %f, precision %f, recall %f, f1 %f, yes_ratio %f" % (
    #     acc_ra, precision_ra, recall_ra, f1_ra, yes_ratio_ra))
    # print("The result of popular analysis is: accuracy %f, precision %f, recall %f, f1 %f, yes_ratio %f" % (
    #     acc_po, precision_po, recall_po, f1_po, yes_ratio_po))
    # print("The result of adversarial analysis is: accuracy %f, precision %f, recall %f, f1 %f, yes_ratio %f" % (
    #     acc_ad, precision_ad, recall_ad, f1_ad, yes_ratio_ad))


    ##### hidden states ######
    # figure_name = "ra"  # 'ad', 'po'
    # for layer in ['layer_0', 'layer_1', 'layer_2', 'layer_3']:
    #     analysis_hidden_states(file_ra, figure_name, layer)


    #### calculate the consistency between different responses #########
    file_ad_1 = "coco2014_val/answer_coco_pope_adversarial_1.json"
    file_ad_2 = "coco2014_val/answer_coco_pope_adversarial_2.json"
    file_ad_3 = "coco2014_val/answer_coco_pope_adversarial_3.json"
    file_ad_4 = "coco2014_val/answer_coco_pope_adversarial_4.json"
    file_ad_5 = "coco2014_val/answer_coco_pope_adversarial_5.json"
    file_ad_6 = "coco2014_val/answer_coco_pope_adversarial_6.json"
    file_ad_7 = "coco2014_val/answer_coco_pope_adversarial_7.json"
    file_ad_8 = "coco2014_val/answer_coco_pope_adversarial_8.json"
    file_ad_9 = "coco2014_val/answer_coco_pope_adversarial_9.json"
    res = calculate_consistency_between_respones(file_ad_1, file_ad_2, file_ad_3, file_ad_4, file_ad_5, file_ad_6, file_ad_7, file_ad_8, file_ad_9)
    print(res)

    # hidden_states = []
    # labels = []
    # texts = []
    # prompts = []
    # qus = []
    # with open(file_ad, "r") as f:
    #     for line in f.readlines():
    #         dic = json.loads(line)
    #         text = dic['text']
    #         label = dic['label']
    #         prompt = dic['prompt']
    #         if dic['text'].lower() != dic['label'].lower():
    #             hidden_states.append(dic['layer_0'])
    #             labels.append(label)
    #             texts.append(text)
    #             prompts.append(prompt)
    #             qus.append(dic['question_id'])



















