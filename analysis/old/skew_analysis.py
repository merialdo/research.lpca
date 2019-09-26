import numpy as np
import matplotlib.pyplot as plt
from io_utils import *


def plot_dict(dictionary, title, xlabel, ylabel, xticks):
    x = []
    y = []

    items = sorted(dictionary.items(), key=lambda x: x[1])

    for i in range(len(items)):
        item = items[i]
        x.append(i)
        y.append(item[1])
    plt.plot(x, y, '.-', lw=2, markersize=2)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.yscale('log')
    plt.xscale('log')


    plt.grid(True)


    plt.show()



def plot_dicts(dictionary1, dictionary2, title, xlabel, ylabel, xticks):
    x1 = []
    y1 = []
    items1 = sorted(dictionary1.items(), key=lambda x: x[1])
    for i in range(len(items1)):
        item = items1[i]
        x1.append(i)
        y1.append(item[1])
    plt.plot(x1, y1, '-', lw=2)

    x2 = []
    y2 = []
    items2 = sorted(dictionary2.items(), key=lambda x: x[1])
    for i in range(len(items2)):
        item = items2[i]
        x2.append(i)
        y2.append(item[1])
    plt.plot(x2, y2, '-', lw=2)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.yscale('log')
    plt.xscale('log')


    plt.grid(True)


    plt.show()


fb15k_entity_degrees, fb15k_relation_mentions = get_degrees_from_openke_files(["/Users/andrea/paper/FB15K/train2id.txt", "/Users/andrea/paper/FB15K/test2id.txt"])
wn18_entity_degrees, wn18_relation_mentions = get_degrees_from_openke_files(["/Users/andrea/paper/WN18/train2id.txt", "/Users/andrea/paper/WN18/test2id.txt"])
fb15k237_entity_degrees, fb15k237_relation_mentions = get_degrees_from_openke_files(["/Users/andrea/paper/FB15K237/train2id.txt", "/Users/andrea/paper/FB15K237/test2id.txt"])
wn18rr_entity_degrees, wn18rr_relation_mentions = get_degrees_from_openke_files(["/Users/andrea/paper/WN18RR/train2id.txt", "/Users/andrea/paper/WN18RR/test2id.txt"])

#plot_dict(fb15k_entity_degrees, "FB15K entity degrees", "Entity", " Entity degree", xticks=[0, 3000, 6000, 9000, 12000])
#plot_dict(fb15k_relation_mentions, "FB15K relation mentions", "Relation", " Relation mentions", xticks=[0, 300, 600, 900, 1200])
#plot_dict(wn18_entity_degrees, "WN18 entity degrees", "Entity", " Entity degree", xticks=[0, 9000, 18000, 27000, 36000])
#plot_dict(wn18_relation_mentions, "WN18 relation mentions", "Relation", " Relation mentions", xticks=[0, 4, 8, 12, 16])
#plot_dict(fb15k237_entity_degrees, "FB15K-237 entity degrees", "Entity", " Entity degree", xticks=[0, 3000, 6000, 9000, 12000])
#plot_dict(fb15k237_relation_mentions, "FB15K-237 relation mentions", "Relation", " Relation mentions", xticks=[0, 50, 100, 150, 200])
#plot_dict(wn18rr_entity_degrees, "WN18-RR entity degrees", "Entity", " Entity degree", xticks=[0, 9000, 18000, 27000, 36000])
#plot_dict(wn18rr_relation_mentions, "WN18-RR relation mentions", "Relation", " Relation mentions", xticks=[0, 2, 4, 6, 8])


plot_dicts(fb15k_entity_degrees, wn18_entity_degrees, "FB15K vs WN18 entity degrees", "Entity", " Entity degree", xticks=[0, 3000, 6000, 9000, 12000])
plot_dicts(fb15k_relation_mentions, wn18_relation_mentions, "FB15K vs WN18 relation mentions", "Relation", "Relation mentions", xticks=[0, 300, 600, 900, 1200])

