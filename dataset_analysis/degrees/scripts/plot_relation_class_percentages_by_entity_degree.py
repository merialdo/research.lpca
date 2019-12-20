import numpy as np
import matplotlib.pyplot as plt

from dataset_analysis.degrees import relation_mentions, entity_degrees
from dataset_analysis.relation_cardinalities import relation_coarse_classes
from datasets import FB15K, Dataset
from io_utils import *
from collections import defaultdict


def plot(x, y, title, xlabel, ylabel):
    plt.scatter(x, y, s=1, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


entity_2_in_degree, entity_2_out_degree, entity_2_degree,= entity_degrees.read(FB15K)
relation_2_mentions = relation_mentions.read(FB15K)
relation_2_coarse_class = relation_coarse_classes.read(FB15K, return_rel_2_class=True)

# for each entity, compute a dict "type of relation" -> number of times that the entity occurs as head for that type of relation
entity_2_head_counts = defaultdict(lambda: defaultdict(lambda: 0))

# for each entity, compute a dict "type of relation" -> number of times that the entity occurs as tail for that type of relation
entity_2_tail_counts = defaultdict(lambda: defaultdict(lambda: 0))

dataset = Dataset(FB15K)

for (head, relation, tail) in dataset.train_triples:
    entity_2_head_counts[head][relation_2_coarse_class[relation]] += 1
    entity_2_tail_counts[tail][relation_2_coarse_class[relation]] += 1


# for each entity, compute a dict
#   "type of relation" -> percentage times that the entity occurs as head for that type of relation
#                           (in relation to the times that it occurs as head overall)
entity_2_head_percentages = defaultdict(lambda: defaultdict(lambda: 0))

# for each entity, compute a dict
#   "type of relation" -> percentage times that the entity occurs as tail for that type of relation
#                           (in relation to the times that it occurs as tail overall)
entity_2_tail_percentages = defaultdict(lambda: defaultdict(lambda: 0))

for entity in entity_2_head_counts:
    all = 0
    for relation_type in entity_2_head_counts[entity]:
        all += entity_2_head_counts[entity][relation_type]
    for relation_type in entity_2_head_counts[entity]:
        entity_2_head_percentages[entity][relation_type] = entity_2_head_counts[entity][relation_type]*100/all

for entity in entity_2_tail_counts:
    all = 0
    for relation_type in entity_2_tail_counts[entity]:
        all += entity_2_tail_counts[entity][relation_type]
    for relation_type in entity_2_tail_counts[entity]:
        entity_2_tail_percentages[entity][relation_type] = entity_2_tail_counts[entity][relation_type]*100/all


# sort all entities by in degree
entities_sorted_by_in_degree = sorted(entity_2_head_counts.keys(), key=lambda x:entity_2_in_degree[x])

# sort all entities by out degree
entities_sorted_by_out_degree = sorted(entity_2_head_counts.keys(), key=lambda x:entity_2_out_degree[x])


in_degrees = []
y_1_to_1_tail, y_1_to_n_tail, y_n_to_1_tail, y_n_to_n_tail = [], [], [], []
y_1_to_1_head, y_1_to_n_head, y_n_to_1_head, y_n_to_n_head = [], [], [], []

for entity in entities_sorted_by_in_degree:
    in_degrees.append(entity_2_in_degree[entity])

    y_1_to_1_head.append(entity_2_head_percentages[entity]["one to one"])
    y_1_to_1_tail.append(entity_2_tail_percentages[entity]["one to one"])

    y_1_to_n_head.append(entity_2_head_percentages[entity]["one to many"])
    y_1_to_n_tail.append(entity_2_tail_percentages[entity]["one to many"])

    y_n_to_1_head.append(entity_2_head_percentages[entity]["many to one"])
    y_n_to_1_tail.append(entity_2_tail_percentages[entity]["many to one"])

    y_n_to_n_head.append(entity_2_head_percentages[entity]["many to many"])
    y_n_to_n_tail.append(entity_2_tail_percentages[entity]["many to many"])


plot(in_degrees, y_1_to_1_head, "In-degree of each entity vs its percentage of one-to-one head predictions", "in-degree", "% of occurrences of that entity as head of one to one relations")
plot(in_degrees, y_1_to_1_tail, "In-degree of each entity vs its percentage of one-to-one tail predictions", "in-degree", "% of occurrences of that entity as tail of one to one relations")

plot(in_degrees, y_1_to_n_head, "In-degree of each entity vs its percentage of one-to-many head predictions", "in-degree", "% of occurrences of that entity as head of one to many relations")
plot(in_degrees, y_1_to_n_tail, "In-degree of each entity vs its percentage of one-to-many tail predictions", "in-degree", "% of occurrences of that entity as tail of one to many relations")

plot(in_degrees, y_n_to_1_head, "In-degree of each entity vs its percentage of many-to-one head predictions", "in-degree", "% of occurrences of that entity as head of many to one relations")
plot(in_degrees, y_n_to_1_tail, "In-degree of each entity vs its percentage of many-to-one tail predictions", "in-degree", "% of occurrences of that entity as tail of many to one relations")

plot(in_degrees, y_n_to_n_head, "In-degree of each entity vs its percentage of many-to-many head predictions", "in-degree", "% of occurrences of that entity as head of many to many relations")
plot(in_degrees, y_n_to_n_tail, "In-degree of each entity vs its percentage of many-to-many tail predictions", "in-degree", "% of occurrences of that entity as tail of many to many relations")






out_degrees = []
y_1_to_1_tail, y_1_to_n_tail, y_n_to_1_tail, y_n_to_n_tail = [], [], [], []
y_1_to_1_head, y_1_to_n_head, y_n_to_1_head, y_n_to_n_head = [], [], [], []

for entity in entities_sorted_by_out_degree:
    out_degrees.append(entity_2_in_degree[entity])
    y_1_to_1_tail.append(entity_2_tail_percentages[entity]["one to one"])
    y_1_to_1_head.append(entity_2_head_percentages[entity]["one to one"])

    y_1_to_n_tail.append(entity_2_tail_percentages[entity]["one to many"])
    y_1_to_n_head.append(entity_2_head_percentages[entity]["one to many"])

    y_n_to_1_tail.append(entity_2_tail_percentages[entity]["many to one"])
    y_n_to_1_head.append(entity_2_head_percentages[entity]["many to one"])

    y_n_to_n_tail.append(entity_2_tail_percentages[entity]["many to many"])
    y_n_to_n_head.append(entity_2_head_percentages[entity]["many to many"])


plot(in_degrees, y_1_to_1_head, "Out-degree of each entity vs its percentage of one-to-one head predictions", "Out-degree", "% of occurrences of that entity as head of one to one relations")
plot(in_degrees, y_1_to_1_tail, "Out-degree of each entity vs its percentage of one-to-one tail predictions", "Out-degree", "% of occurrences of that entity as tail of one to one relations")

plot(in_degrees, y_1_to_n_head, "Out-degree of each entity vs its percentage of one-to-many head predictions", "Out-degree", "% of occurrences of that entity as head of one to many relations")
plot(in_degrees, y_1_to_n_tail, "Out-degree of each entity vs its percentage of one-to-many tail predictions", "Out-degree", "% of occurrences of that entity as tail of one to many relations")

plot(in_degrees, y_n_to_1_head, "Out-degree of each entity vs its percentage of many-to-one head predictions", "Out-degree", "% of occurrences of that entity as head of many to one relations")
plot(in_degrees, y_n_to_1_tail, "Out-degree of each entity vs its percentage of many-to-one tail predictions", "Out-degree", "% of occurrences of that entity as tail of many to one relations")

plot(in_degrees, y_n_to_n_head, "Out-degree of each entity vs its percentage of many-to-many head predictions", "Out-degree", "% of occurrences of that entity as head of many to many relations")
plot(in_degrees, y_n_to_n_tail, "Out-degree of each entity vs its percentage of many-to-many tail predictions", "Out-degree", "% of occurrences of that entity as tail of many to many relations")