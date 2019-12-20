import numpy
import matplotlib.pyplot as plt

from dataset_analysis.relation_cardinalities import relation_cardinalities
from datasets import Dataset, FB15K


def plot(x, y, title, xlabel, ylabel):
    plt.scatter(x, y, s=1, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter([range(500)], range(500), s=1, color='red')

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


one_2_one = []
one_2_many = []
many_2_one = []
many_2_many = []

dataset = Dataset(FB15K)
rel_2_cardinality_dicts = relation_cardinalities.compute(dataset)



many_2_many_dict = dict()

for rel in rel_2_cardinality_dicts:
    head_2_tails = rel_2_cardinality_dicts[rel]["head_to_tails"]
    avg_tails_for_head = numpy.average([len(head_2_tails[head]) for head in head_2_tails])

    tail_2_heads = rel_2_cardinality_dicts[rel]["tail_to_heads"]
    avg_heads_for_tail = numpy.average([len(tail_2_heads[tail]) for tail in tail_2_heads])

    if avg_tails_for_head <= 1.5 and avg_heads_for_tail <= 1.5:
        one_2_one.append(rel)
    elif avg_tails_for_head <= 1.5 and avg_heads_for_tail > 1.5:
        many_2_one.append(rel)
    elif avg_tails_for_head > 1.5 and avg_heads_for_tail <= 1.5:
        one_2_many.append(rel)
    else:
        many_2_many.append(rel)
        many_2_many_dict[rel] = (avg_tails_for_head, avg_heads_for_tail)

x = []
y = []

for relationship in many_2_many_dict.keys():
    avg_tails_for_head, avg_heads_for_tail = many_2_many_dict[relationship]
    x.append(avg_tails_for_head)
    y.append(avg_heads_for_tail)


plot(x, y, "many to many analysis", "average tails a parità di head", "average heads a parità di tail")
