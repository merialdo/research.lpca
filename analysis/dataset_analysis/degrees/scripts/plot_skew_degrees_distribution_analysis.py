import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

from dataset_analysis.degrees import entity_degrees, relation_mentions
from datasets import FB15K, WN18, FB15K_237, WN18RR
from io_utils import *


def plot_dict(dictionary, title, xlabel, ylabel, xticks):
    x = []
    y = []

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    items = sorted(dictionary.items(), key=lambda x: x[0])
    for i in range(len(items)):
        (count, amount_of_elements_with_that_count) = items[i]
        x.append(int(count))
        y.append(int(amount_of_elements_with_that_count))

    plt.scatter(x, y, s=6, label="FB15K")
    plt.legend(markerscale=2., scatterpoints=1, fontsize=18)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.yscale('log')
    plt.xscale('log')

    plt.grid(True)
    plt.show()

def plot_dicts(fb15k_dict, wn18_dict, title, xlabel, ylabel, xticks):
    x1 = []
    y1 = []


    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    fb15k_items = sorted(fb15k_dict.items(), key=lambda x: x[1])
    for i in range(len(fb15k_items)):
        (count, amount_of_elements_with_that_count) = fb15k_items[i]
        x1.append(int(count))
        y1.append(int(amount_of_elements_with_that_count))
    plt.scatter(x1, y1, s=3, label="FB15K")
    plt.legend(markerscale=4., scatterpoints=1, fontsize=18)

    x2 = []
    y2 = []
    wn18_items = sorted(wn18_dict.items(), key=lambda x: x[1])
    for i in range(len(wn18_items)):
        (count, amount_of_elements_with_that_count) = wn18_items[i]
        x2.append(int(count))
        y2.append(int(amount_of_elements_with_that_count))
    plt.scatter(x2, y2, s=3, label="WN18")
    plt.legend(markerscale=4., scatterpoints=1, fontsize=18)

    #plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks([])
    plt.xscale('log')
    plt.yscale('log')


    plt.grid(True)


    plt.show()



def distribution_of(name_2_count):
    count_2_amount_of_names = defaultdict(lambda:0)

    for (name, count) in name_2_count.items():
        count_2_amount_of_names[count] += 1

    return count_2_amount_of_names


_, _, fb15k_entity_degrees = entity_degrees.read(FB15K)
fb15k_relation_mentions = relation_mentions.read(FB15K)

_, _, wn18_entity_degrees = entity_degrees.read(WN18)
wn18_relation_mentions = relation_mentions.read(WN18)

_, _, fb15k237_entity_degrees = entity_degrees.read(FB15K_237)
fb15k237_relation_mentions = relation_mentions.read(FB15K_237)

_, _, wn18rr_entity_degrees = entity_degrees.read(WN18RR)
wn18rr_relation_mentions = relation_mentions.read(WN18RR)


fb15k_entity_degrees_distribution = distribution_of(fb15k_entity_degrees)
fb15k_relation_mentions_distribution = distribution_of(fb15k_relation_mentions)

wn18_entity_degrees_distribution = distribution_of(wn18_entity_degrees)
wn18_relation_mentions_distribution = distribution_of(wn18_relation_mentions)

fb15k237_entity_degrees_distribution = distribution_of(fb15k237_entity_degrees)
fb15k237_relation_mentions_distribution = distribution_of(fb15k237_relation_mentions)

wn18rr_entity_degrees_distribution = distribution_of(wn18rr_entity_degrees)
wn18rr_relation_mentions_distribution = distribution_of(wn18rr_relation_mentions)

print(wn18rr_relation_mentions_distribution)
#plot_dict(fb15k_entity_degrees_distribution, "FB15K entity degrees", "Entity degree", "Amount of entities with that degree", xticks=[0, 3000, 6000, 9000, 12000])
#plot_dict(fb15k_relation_mentions_distribution, "FB15K relation mentions", "Relation mentions", " Amount of relations with those mentions", xticks=[0, 300, 600, 900, 1200])
#plot_dict(wn18_entity_degrees_distribution, "WN18 entity degrees", "Entity degree", " Amount of entities with that degree", xticks=[0, 9000, 18000, 27000, 36000])
#plot_dict(wn18_relation_mentions_distribution, "WN18 relation mentions", "Relation mentions", " Amount of relations with those mentions", xticks=[0, 4, 8, 12, 16])
#plot_dict(fb15k237_entity_degrees_distribution, "FB15K-237 entity degrees", "Entity degree", " Amount of entities with that degree", xticks=[0, 3000, 6000, 9000, 12000])
#plot_dict(fb15k237_relation_mentions_distribution, "FB15K-237 relation mentions", "Relation mentions", " Amount of relations with those mentions", xticks=[0, 300, 600, 900, 1200])
#plot_dict(wn18rr_entity_degrees_distribution, "WN18-RR entity degrees", "Entity degree", " Amount of entities with that degree", xticks=[0, 9000, 18000, 27000, 36000])
#plot_dict(wn18rr_relation_mentions_distribution, "WN18-RR relation mentions", "Relation mentions", " Amount of relations with those mentions", xticks=[0, 4, 8, 12, 16])

plot_dicts(fb15k_entity_degrees_distribution, wn18_entity_degrees_distribution, "FB15K vs WN18 entity degrees distribution", "Entity degree", "Number of entities", xticks=[])
plot_dict(fb15k_relation_mentions_distribution, "Relationship mentions distribution in FB15K", "Number of mentions for a relationship", "Number of relationships", xticks=[])

