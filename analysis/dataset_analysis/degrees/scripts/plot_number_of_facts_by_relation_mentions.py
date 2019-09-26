import matplotlib.pyplot as plt

from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from datasets import FB15K, Dataset, YAGO3_10

from io_utils import *
from collections import defaultdict

rel_2_mentions = relation_mentions.read(FB15K)

def plot_dict(dict, title, xlabel, ylabel):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, s=10, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


items = []
for item in sorted(rel_2_mentions.items(), key=lambda x:x[1], reverse=True):
    items.append(item)

x = []
y = []
for i in range(len(items)):
    item = items[i]
    x.append(i)
    y.append(item[1])

index_mediana = int((len(items)+1)/2)

item_mediano = items[index_mediana]

mentions_media = 0
for item in items:
    mentions_media += item[1]
mentions_media = mentions_media/len(items)

print("MEDIA: ")
print(mentions_media)
print("MEDIANA: ")
print(item_mediano[1])

plt.scatter(x, y, s=1, color='blue')

    #plt.xscale('log')
    #plt.yscale('log')
plt.show()


