import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.interpolate import interp1d

import performances
from datasets import FB15K
from io_utils import *

from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from models import TRANSE, ROTATE, CONVE, SIMPLE, ANYBURL


def plot_dicts(dicts, title, xlabel, ylabel):

    for dict in dicts:
        x = []
        y = []

        for item in (sorted(dict.items(), key=lambda x: x[0])):
            x.append(item[0])
            y.append(item[1])


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


def get_dicts_from_entries(entries, entity2degree, relation2degree):
    entity_degree_2_ranks = defaultdict(lambda: [])
    relation_mentions_2_ranks = defaultdict(lambda: [])

    for entry in entries:
        head_degree = int(entity2degree[entry["head"]])
        tail_degree = int(entity2degree[entry["tail"]])
        relation_mentions = int(relation2degree[entry["relation"]])
        head_rank = int(entry["head_rank_filtered"])
        tail_rank = int(entry["tail_rank_filtered"])

        entity_degree_2_ranks[head_degree].append(head_rank)
        entity_degree_2_ranks[tail_degree].append(tail_rank)

        relation_mentions_2_ranks[relation_mentions].append(head_rank)
        relation_mentions_2_ranks[relation_mentions].append(tail_rank)

    entity_degree_2_mean_rank = dict()
    relation_mentions_2_mean_rank = dict()

    for item in entity_degree_2_ranks.items():
        if len(item[1]) > 100:
            entity_degree_2_mean_rank [item[0]] = np.average(item[1])
        else:
            entity_degree_2_mean_rank[item[0]] = None

    for item in relation_mentions_2_ranks.items():

        relation_mentions_2_mean_rank[item[0]] = np.average(item[1])

    return entity_degree_2_mean_rank, relation_mentions_2_mean_rank


_, _, entity2degree = entity_degrees.read(FB15K)
relation2mentions = relation_mentions.read(FB15K)

transE_entries = performances.read_filtered_ranks_entries_for(TRANSE, FB15K)
rotatE_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)
convE_entries = performances.read_filtered_ranks_entries_for(CONVE, FB15K)
simplE_entries = performances.read_filtered_ranks_entries_for(SIMPLE, FB15K)
anyburl_entries = performances.read_filtered_ranks_entries_for(ANYBURL, FB15K)

transE_entity_degree_2_rank, transE_relation_mentions_2_rank = get_dicts_from_entries(transE_entries, entity2degree, relation2mentions)
rotatE_entity_degree_2_rank, rotatE_relation_mentions_2_rank = get_dicts_from_entries(rotatE_entries, entity2degree, relation2mentions)
convE_entity_degree_2_rank, convE_relation_mentions_2_rank = get_dicts_from_entries(convE_entries, entity2degree, relation2mentions)
simplE_entity_degree_2_rank, simplE_relation_mentions_2_rank = get_dicts_from_entries(simplE_entries, entity2degree, relation2mentions)
anyburl_entity_degree_2_rank, anyburl_relation_mentions_2_rank = get_dicts_from_entries(anyburl_entries, entity2degree, relation2mentions)


plot_dict(transE_entity_degree_2_rank, "TransE entity degree vs mean rank", "degree", "avg rank of correct predictions of entities with that degree")
#plot_dict(transE_entity_degree_2_rank, "TransE relation mentions vs mean rank", "relation mentions", "avg rank of correct head for test facts with that relation mentions")

plot_dict(rotatE_entity_degree_2_rank, "RotatE entity degree vs mean rank", "degree", "avg rank of correct predictions of entities with that degree")
#plot_dict(rotatE_relation_mentions_2_rank, "RotatE relation mentions vs mean rank", "relation mentions", "avg rank of correct head for test facts with that relation mentions")

plot_dict(convE_entity_degree_2_rank, "ConvE entity degree vs mean rank", "degree", "avg rank of correct predictions of entities with that degree")
#plot_dict(convE_relation_mentions_2_rank, "ConvE relation mentions vs mean rank", "relation mentions", "avg rank of correct head for test facts with that relation mentions")

plot_dict(simplE_entity_degree_2_rank, "SimplE entity degree vs mean rank", "degree", "avg rank of correct predictions of entities with that degree")
#plot_dict(simplE_relation_mentions_2_rank, "SimplE relation mentions vs mean rank", "relation mentions", "avg rank of correct head for test facts with that relation mentions")

plot_dict(anyburl_entity_degree_2_rank, "AnyBURL entity degree vs mean rank", "degree", "avg rank of correct predictions of entities with that degree")
#plot_dict(anyburl_relation_mentions_2_rank, "AnyBURL relation mentions vs mean rank", "relation mentions", "avg rank of correct head for test facts with that relation mentions")