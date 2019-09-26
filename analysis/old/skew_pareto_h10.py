import numpy as np
import matplotlib.pyplot as plt
from io_utils import *
from collections import defaultdict


def plot_dict(dict, title, xlabel, ylabel, xticks=[]):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, s=1)
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)


    plt.grid(True)


    plt.show()


def plot_dicts(dict1, dict2, title, xlabel, ylabel, xticks=[]):
    x1 = []
    y1 = []
    for item in (sorted(dict1.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])

    plt.scatter(x1, y1, s=1)

    x2 = []
    y2 = []
    for item in (sorted(dict2.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])

    plt.scatter(x2, y2, s=1)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.xscale('log')


    plt.grid(True)


    plt.show()


def get_degrees_from_entries(entries):
    entity_2_degree = dict()

    for entry in entries:
        entity_2_degree[entry["head"]] = entry["head_degree"]
        entity_2_degree[entry["tail"]] = entry["tail_degree"]

    return entity_2_degree

def get_questions_from_entries(entries):

    questions = []

    for entry in entries:

        question = dict()
        question["entity"] = entry["head"]
        question["degree"] = entry["head_degree"]
        question["rank"] = entry["head_rank_raw"]
        questions.append(question)

        question = dict()
        question["entity"] = entry["tail"]
        question["degree"] = entry["tail_degree"]
        question["rank"] = entry["tail_rank_raw"]
        questions.append(question)

    return questions


def get_global_results_for_questions_skipping_topk_entities(sorted_entity_2_degree_items, sorted_questions, k):

    all_ranks = []
    hits_at_10 = 0.0

    allowed_entities = set([item[0] for item in sorted_entity_2_degree_items[k:]])
    allowed_questions = 0.0

    for question in sorted_questions:
        if question["entity"] in allowed_entities:
            allowed_questions += 1
            all_ranks.append(question["rank"])
            if question['rank'] <= 10:
                hits_at_10 += 1

    return np.average(all_ranks), hits_at_10/allowed_questions




entries = get_entries_from_file("/Users/andrea/paper/WN18/distmult_wn18_test_with_correct_ranks.csv")

questions = get_questions_from_entries(entries)

entity_2_hits10 = defaultdict(lambda:0)

for question in questions:
    if question['rank'] <= 10:
        entity_2_hits10[question['entity']] += 1

sorted_entity_2_hits10 = sorted(entity_2_hits10.items(), key=lambda item: item[1], reverse=True)

total = 0
for item in sorted_entity_2_hits10:
    total+=item[1]

erased_h10 = 0
erased_entities = []
threshold = (total/100)*80
while(erased_h10 < threshold):
    to_erase = sorted_entity_2_hits10[0]

    sorted_entity_2_hits10 = sorted_entity_2_hits10[1:]

    erased_entities.append(to_erase[0])
    erased_h10 += to_erase[1]

print(len(erased_entities))
# distmult_entries = get_test_prediction_entries_from("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")




