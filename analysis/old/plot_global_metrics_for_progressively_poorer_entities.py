import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from bucket import bucket
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


def poly_func(x, coeffs):
    result = 0
    for i in range(len(coeffs)):
        coeff = coeffs[i]
        exp = len(coeffs) - 1 - i
        result += pow(x, exp) * coeff
    return result

def plot_dicts(dict1, dict2, title, xlabel, ylabel, xticks=[]):

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    x1 = []
    y1 = []
    for item in (sorted(dict1.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])
    x1 = x1[:int(len(x1)*95/100)]
    y1 = y1[:int(len(y1)*95/100)]

    #coeffs = np.polyfit(x1, y1, 4)
    #y1_polyfit = [min(poly_func(x, coeffs), 1) for x in x1]
    #plt.plot(x1, y1_polyfit, '--', label='TransE approx.')
    #plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    #plt.scatter(x1, y1, s=1, label="TransE")
    #x1, y1 = bucket(x1, y1, 15)

    plt.plot(x1, y1, label='TransE')
    #plt.scatter(x1, y1, marker='x', s=30, label='TransE')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)


    x2 = []
    y2 = []
    for item in (sorted(dict2.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])
    x2 = x2[:int(len(x2)*95/100)]
    y2 = y2[:int(len(y2)*95/100)]

    #coeffs = np.polyfit(x2, y2, 4)
    #y2_polyfit = [min(poly_func(x, coeffs), 1) for x in x2]
    #plt.plot(x2, y2_polyfit, '--', label='DistMult approx.')
    #plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    #x2, y2 = bucket(x2, y2, 15)
    plt.plot(x2, y2, label='DistMult')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

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

def get_dicts_from_entries(entries):
    k_2_h10 = defaultdict(lambda:0)
    k_2_meanrank = defaultdict(lambda:0)
    for entry in entries:
        k_2_h10[entry['k']] = entry['h10']
        k_2_meanrank[entry['k']] = entry['mean_rank']

    return k_2_h10, k_2_meanrank


transe_entries = get_entries_from_progressively_poorer_entities_from_file("/Users/andrea/paper/FB15K/transe_progressively_skipping_top_k_entities.csv")
distmult_entries = get_entries_from_progressively_poorer_entities_from_file("/Users/andrea/paper/FB15K/distmult_progressively_skipping_top_k_entities.csv")

transe_k_2_h10, transe_k_2_meanrank = get_dicts_from_entries(transe_entries)
distmult_k_2_h10, distmult_k_2_meanrank = get_dicts_from_entries(distmult_entries)

plot_dicts(transe_k_2_h10, distmult_k_2_h10, "Skipped entities vs global Hits@10", "Amount of top entities skipped", "Global Hits@10")
plot_dicts(transe_k_2_meanrank, distmult_k_2_meanrank, "Skipped entities vs global Mean Rank", "Amount of top entities skipped", "Global Mean Rank")


# distmult_entries = get_test_prediction_entries_from("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")




