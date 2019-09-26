import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from datasets import FB15K, YAGO3_10

from io_utils import *
from models import TRANSE, ROTATE, SIMPLE, CONVE, ANYBURL


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


def get_dicts_from_entries(entries, entity2degree, relation2mentions):
    entity_degree_2_hits = defaultdict(lambda: 0.0)
    relation_mentions_2_hits = defaultdict(lambda: 0.0)
    entity_degree_2_count = defaultdict(lambda: 0.0)
    relation_mentions_2_count = defaultdict(lambda: 0.0)

    for entry in entries:
        head_degree = int(entity2degree[entry["head"]])
        tail_degree = int(entity2degree[entry["tail"]])
        relation_mentions_number = int(relation2mentions[entry["relation"]])
        head_rank = int(entry["head_rank_filtered"])
        tail_rank = int(entry["tail_rank_filtered"])

        entity_degree_2_count[head_degree] += 1.0
        entity_degree_2_count[tail_degree] += 1.0
        relation_mentions_2_count[relation_mentions_number] += 1.0

        if head_rank == 1:
            entity_degree_2_hits[head_degree] += 1.0
            relation_mentions_2_hits[relation_mentions_number] += 1.0

        if tail_rank == 1:
            entity_degree_2_hits[tail_degree] += 1.0
            relation_mentions_2_hits[relation_mentions_number] += 1.0

    for key in entity_degree_2_hits:
        entity_degree_2_hits[key] = entity_degree_2_hits[key]/entity_degree_2_count[key]
    for key in relation_mentions_2_hits:
        relation_mentions_2_hits[key] = relation_mentions_2_hits[key]/relation_mentions_2_count[key]

    return entity_degree_2_hits, relation_mentions_2_hits

dataset_name = YAGO3_10
models_names = [ROTATE]

_, _, entity_2_degree = entity_degrees.read(dataset_name)
relation_2_mentions = relation_mentions.read(dataset_name)

for model_name in models_names:
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name)
    model_entity_degree_2_hits, model_relation_mentions_2_hits = get_dicts_from_entries(model_entries,
                                                                                          entity_2_degree,
                                                                                          relation_2_mentions)

    plot_dict(model_entity_degree_2_hits, model_name + " entity degree vs hits@1", "degree", "percentage of hits@1 on all predictions of entities with that degree")
    # plot_dict(rotatE_relation_mentions_2_hits, "RotatE relation mentions vs mean rank", "relation mentions", "percentage of hits@1 on all predictions of entities with that degree")
