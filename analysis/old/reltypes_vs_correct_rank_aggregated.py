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


def plot_dicts(dict1, dict2, title, xlabel, ylabel):

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    x1 = []
    y1 = []

    for item in (sorted(dict1.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])

    plt.scatter(x1, y1, marker='.', label='TransE')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x2 = []
    y2 = []

    for item in (sorted(dict2.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])

    plt.scatter(x2, y2, marker='.', s=30, label='DistMult')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    # plt.xscale('log')
    # plt.yscale('log')
    plt.show()




def get_aggregated_dicts_from_entries(entries, mid_2_reltypes):
    entity_reltypes_2_rank_same = dict()
    entity_reltypes_2_rank_other = dict()

    for entry in entries:
        head_reltypes = mid_2_reltypes[entry["head"]]
        tail_reltypes = mid_2_reltypes[entry["tail"]]
        head_rank = int(entry["head_rank_raw"])
        tail_rank = int(entry["tail_rank_raw"])


        if head_reltypes in entity_reltypes_2_rank_same:
            entity_reltypes_2_rank_same[head_reltypes].append(head_rank)
        else:
            entity_reltypes_2_rank_same[head_reltypes] = [head_rank]

        if tail_reltypes in entity_reltypes_2_rank_same:
            entity_reltypes_2_rank_same[tail_reltypes].append(tail_rank)
        else:
            entity_reltypes_2_rank_same[tail_reltypes] = [tail_rank]


        if head_reltypes in entity_reltypes_2_rank_other:
            entity_reltypes_2_rank_other[head_reltypes].append(tail_rank)
        else:
            entity_reltypes_2_rank_other[head_reltypes] = [tail_rank]

        if tail_reltypes in entity_reltypes_2_rank_other:
            entity_reltypes_2_rank_other[tail_reltypes].append(head_rank)
        else:
            entity_reltypes_2_rank_other[tail_reltypes] = [head_rank]


    entity_reltypes_2_avg_rank_same = dict()
    entity_reltypes_2_avg_rank_other = dict()

    for item in entity_reltypes_2_rank_same.items():
        entity_reltypes_2_avg_rank_same[item[0]] = np.average(item[1])

    for item in entity_reltypes_2_rank_other.items():
        entity_reltypes_2_avg_rank_other[item[0]] = np.average(item[1])

    return entity_reltypes_2_avg_rank_same, entity_reltypes_2_avg_rank_other


mid_2_reltypes = {}

with open("/Users/andrea/paper/FB15K/entity_reltypes.csv") as input_file:
    for line in input_file.readlines():
        mid, reltypes = line.strip().split(";")
        mid_2_reltypes[mid] = int(reltypes)

transe_entries = get_entries_from_file("/Users/andrea/paper/FB15K/transe_fb15k_test_with_correct_ranks.csv")
distmult_entries = get_entries_from_file("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")

transe_same, transe_other = get_aggregated_dicts_from_entries(transe_entries, mid_2_reltypes)
distmult_same, distmult_other = get_aggregated_dicts_from_entries(distmult_entries, mid_2_reltypes)

#plot_dict(transe_same, "TransE entity degree vs entity rank", "entity degree", "avg rank of that entities with that degree in for test facts")
#plot_dict(transe_other, "TransE entity degree vs other entity rank", "entity degree", "avg rank of the other entity in test facts when an entity has that degree")
#plot_dict(transe_rel2ranks, "TransE relation mentions vs entity ranks", "relation mentions", "avg rank of the entities in test facts when the relation has those mentions")

#plot_dict(distmult_same, "DistMult entity degree vs entity rank", "entity degree", "avg rank of that entities with that degree in for test facts")
#plot_dict(distmult_other, "DistMult entity degree vs other entity rank", "entity degree", "avg rank of the other entity in test facts when an entity has that degree")
#plot_dict(distmult_rel2ranks, "DistMult relation mentions vs entity ranks", "relation mentions", "avg rank of the entities in test facts when the relation has those mentions")

plot_dicts(transe_same, distmult_same,  "TransE vs Distmult: entity degree vs entity rank", "Entity relation types", "Avg Rank among predictions")
plot_dicts(transe_other, distmult_other,  "TransE vs Distmult: entity degree vs other entity rank", "entity relation types", "avg rank of the other entity in test facts when an entity has those relation types")


