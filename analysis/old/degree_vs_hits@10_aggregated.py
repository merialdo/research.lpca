import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from io_utils import *
from collections import defaultdict
from bucket import bucket

def plot_dict(dict, title, xlabel, ylabel):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, s=1, color='blue')
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

def plot_dicts(transe_dict, distmult_dict, title, xlabel, ylabel, xticks=[]):

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    x1 = []
    y1 = []
    for item in (sorted(transe_dict.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])

    coeffs = np.polyfit(x1, y1, 4)
    y1_polyfit = [min(poly_func(x, coeffs), 1) for x in x1]
    plt.plot(x1, y1_polyfit, '--', label='TransE approx.')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x1, y1 = bucket(x1, y1, 15)
    plt.scatter(x1, y1, marker='x', s=30, label='TransE')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x2 = []
    y2 = []
    for item in (sorted(distmult_dict.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])

    coeffs = np.polyfit(x2, y2, 4)
    y2_polyfit = [min(poly_func(x, coeffs), 1) for x in x2]

    plt.plot(x2, y2_polyfit, '--', label='DistMult approx.')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    x2, y2 = bucket(x2, y2, 15)
    plt.scatter(x2, y2, marker='x', s=30, label='DistMult')
    plt.legend(markerscale=2., scatterpoints=1, fontsize=14)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xscale('log')


    plt.grid(True)


    plt.show()


def get_h10_dicts_from_entries(entries):

    entity_degree_2_entity_h10 = defaultdict(lambda:0)
    entity_degree_2_other_entity_h10 = defaultdict(lambda:0)
    relation_degree_2_any_h10 = defaultdict(lambda:0)

    entity_degree_2_count = defaultdict(lambda:0)
    relation_mentions_2_count = defaultdict(lambda:0)

    all_entity_degrees = set()
    all_relation_mentions = set()

    for entry in entries:

        head_degree = int(entry["head_degree"])
        tail_degree = int(entry["tail_degree"])
        relation_mentions = int(entry["relation_mentions"])
        head_rank = int(entry["head_rank_raw"])
        tail_rank = int(entry["tail_rank_raw"])

        all_entity_degrees.add(head_degree)
        all_entity_degrees.add(tail_degree)
        all_relation_mentions.add(relation_mentions)

        entity_degree_2_count[head_degree] += 1
        entity_degree_2_count[tail_degree] += 1
        relation_mentions_2_count[relation_mentions] += 1

        if head_rank <= 10:
            entity_degree_2_entity_h10[head_degree] += 1
            entity_degree_2_other_entity_h10[tail_degree] += 1
            relation_degree_2_any_h10[relation_mentions] += 1
        if tail_rank <= 10:
            entity_degree_2_entity_h10[tail_degree] += 1
            entity_degree_2_other_entity_h10[head_degree] += 1
            relation_degree_2_any_h10[relation_mentions] += 1

    entity_degree_2_entity_h10_perc = dict()
    entity_degree_2_other_entity_h10_perc = dict()
    relation_degree_2_any_h10_perc = dict()

    for degree in all_entity_degrees:
        entity_degree_2_entity_h10_perc[degree] = float(entity_degree_2_entity_h10[degree])/float(entity_degree_2_count[degree])
        entity_degree_2_other_entity_h10_perc[degree] = float(entity_degree_2_other_entity_h10[degree])/float(entity_degree_2_count[degree])
    for relation_mentions in all_relation_mentions:
        relation_degree_2_any_h10_perc[relation_mentions] = float(relation_degree_2_any_h10[relation_mentions])/float(relation_mentions_2_count[relation_mentions])

    return entity_degree_2_entity_h10_perc, entity_degree_2_other_entity_h10_perc, relation_degree_2_any_h10_perc


transe_entries = get_entries_from_file("/Users/andrea/paper/FB15K/transe_fb15k_test_with_correct_ranks.csv")
distmult_entries = get_entries_from_file("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")

transe_entity_x_entity, transe_entity_x_other, transe_rel_x_entities = get_h10_dicts_from_entries(transe_entries)
distmult_entity_x_entity, distmult_entity_x_other, distmult_rel_x_entities = get_h10_dicts_from_entries(distmult_entries)



#plot_dict(transe_entity_x_entity, "TransE entity degree vs entity hits@10", "Entity degree", "Avg hits@10 percentage for entities with that degree")
#plot_dict(transe_entity_x_other, "TransE entity degree vs other entity hits@10", "Entity degree", "Avg hits@10 percentage in facts with an entity with that degree, in predicting the other entity")
#plot_dict(transe_rel_x_entities, "TransE relation mentions vs entities hits@10", "Relation mentions", "Avg hits@10 percentage in facts with an entity with a relation with those mentions")

#plot_dict(distmult_entity_x_entity, "DistMult entity degree vs entity hits@10", "Entity degree", "Avg hits@10 percentage for entities with that degree")
#plot_dict(distmult_entity_x_other, "DistMult entity degree vs other entity hits@10", "Entity degree", "Avg hits@10 percentage in facts with an entity with that degree, in predicting the other entity")
#plot_dict(distmult_rel_x_entities, "DistMult relation mentions vs entities hits@10", "Relation mentions", "Avg hits@10 percentage in facts with an entity with a relation with those mentions")

plot_dicts(transe_entity_x_entity, distmult_entity_x_entity, "Entity degree vs average H@10", "Entity degree", "Avg H@10 among predictions")
plot_dicts(transe_entity_x_other, distmult_entity_x_other, "Entity degree vs other entity H@10", "Entity degree", "Avg hits@10 percentage for other entity")