import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

from bucket import bucket
from io_utils import *


def plot_dict(dict, title, xlabel, ylabel):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    x, y = bucket(x, y, 10)
    plt.plot(x, y, label='TransE')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

def poly_func(x, coeffs):
    result = 0
    for i in range(len(coeffs)):
        coeff = coeffs[i]
        exp = len(coeffs) - 1 - i
        result += pow(x, exp) * coeff
    return result

def plot_dicts(dict1, dict2, title, xlabel, ylabel):

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    x1 = []
    y1 = []

    for item in (sorted(dict1.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])

    coeffs = np.polyfit(x1, y1, 4)
    y1_polyfit = [max(poly_func(x, coeffs), 1) for x in x1]
    plt.plot(x1, y1_polyfit, '--', label='TransE approx.')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x1, y1 = bucket(x1, y1, 10)
    plt.scatter(x1, y1, marker='x', s=30, label='TransE')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    #plt.plot(x1, y1, label='TransE')

    x2 = []
    y2 = []
    for item in (sorted(dict2.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])
    #plt.scatter(x2, y2, s=1)
    #coeffs = np.polyfit(x2, y2, 4)
    #plt.plot(x2, [poly_func(x, coeffs) for x in x2], '--', label='DistMult approx.')
    polyfit2 = Polynomial.fit(x2, y2, 4)
    plt.plot(*polyfit2.linspace(),  '--', label='DistMult approx.')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x2, y2 = bucket(x2, y2, 10)
    plt.scatter(x2, y2, marker='x', s=30, label='DistMult')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    #plt.plot(x2, y2, label='DistMult')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.xscale('log')
    plt.yscale('log')
    plt.show()




def get_aggregated_dicts_from_entries(entries):
    entity_degree_2_rank_same = dict()
    entity_degree_2_rank_other = dict()
    relation_mentions_2_rank = dict()

    for entry in entries:
        head_degree = int(entry["head_degree"])
        tail_degree = int(entry["tail_degree"])
        relation_mentions = int(entry["relation_mentions"])
        head_rank = int(entry["head_rank_raw"])
        tail_rank = int(entry["tail_rank_raw"])


        if head_degree in entity_degree_2_rank_same:
            entity_degree_2_rank_same[head_degree].append(head_rank)
        else:
            entity_degree_2_rank_same[head_degree] = [head_rank]

        if tail_degree in entity_degree_2_rank_same:
            entity_degree_2_rank_same[tail_degree].append(tail_rank)
        else:
            entity_degree_2_rank_same[tail_degree] = [tail_rank]


        if head_degree in entity_degree_2_rank_other:
            entity_degree_2_rank_other[head_degree].append(tail_rank)
        else:
            entity_degree_2_rank_other[head_degree] = [tail_rank]
        if tail_degree in entity_degree_2_rank_other:
            entity_degree_2_rank_other[tail_degree].append(head_rank)
        else:
            entity_degree_2_rank_other[tail_degree] = [head_rank]

        if relation_mentions in relation_mentions_2_rank:
            relation_mentions_2_rank[relation_mentions].append(head_rank)
            relation_mentions_2_rank[relation_mentions].append(tail_rank)
        else:
            relation_mentions_2_rank[relation_mentions] = [head_rank, tail_rank]

    entity_degree_2_avg_rank_same = dict()
    entity_degree_2_avg_rank_other = dict()
    relation_mentions_2_avg_rank = dict()

    for item in entity_degree_2_rank_same.items():
        entity_degree_2_avg_rank_same [item[0]] = np.average(item[1])

    for item in entity_degree_2_rank_other.items():
        entity_degree_2_avg_rank_other[item[0]] = np.average(item[1])

    for item in relation_mentions_2_rank.items():
        relation_mentions_2_avg_rank[item[0]] = np.average(item[1])

    return entity_degree_2_avg_rank_same, entity_degree_2_avg_rank_other, relation_mentions_2_avg_rank



transe_entries = get_entries_from_file("/Users/andrea/paper/FB15K/transe_fb15k_test_with_correct_ranks.csv")
distmult_entries = get_entries_from_file("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")

transe_same, transe_other, transe_rel2ranks = get_aggregated_dicts_from_entries(transe_entries)
distmult_same, distmult_other, distmult_rel2ranks = get_aggregated_dicts_from_entries(distmult_entries)


#plot_dict(transe_same, "TransE entity degree vs entity rank", "entity degree", "avg rank of that entities with that degree in for test facts")
#plot_dict(transe_other, "TransE entity degree vs other entity rank", "entity degree", "avg rank of the other entity in test facts when an entity has that degree")
#plot_dict(transe_rel2ranks, "TransE relation mentions vs entity ranks", "relation mentions", "avg rank of the entities in test facts when the relation has those mentions")

#plot_dict(distmult_same, "DistMult entity degree vs entity rank", "entity degree", "avg rank of that entities with that degree in for test facts")
#plot_dict(distmult_other, "DistMult entity degree vs other entity rank", "entity degree", "avg rank of the other entity in test facts when an entity has that degree")
#plot_dict(distmult_rel2ranks, "DistMult relation mentions vs entity ranks", "relation mentions", "avg rank of the entities in test facts when the relation has those mentions")

plot_dicts(transe_same, distmult_same,  "TransE vs Distmult: entity degree vs entity rank", "Entity degree", "Avg Rank among predictions")
#plot_dicts(transe_other, distmult_other,  "TransE vs Distmult: entity degree vs other entity rank", "entity degree", "avg rank of the other entity in test facts when an entity has that degree")


