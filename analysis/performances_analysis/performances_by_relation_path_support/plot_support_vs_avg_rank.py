import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.paths import tfidf_support
from datasets import FB15K, YAGO3_10, Dataset

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


def process(model_entries, test_fact_2_support):

    support_2_head_ranks = defaultdict(lambda: [])
    support_2_tail_ranks = defaultdict(lambda: [])

    for entry in model_entries:

        key = ";".join([entry["head"], entry["relation"], entry["tail"]])

        support = test_fact_2_support[key]

        support_2_head_ranks[support].append(int(entry["head_rank_filtered"]))
        support_2_tail_ranks[support].append(int(entry["tail_rank_filtered"]))

    support_2_head_avg_rank = dict()
    support_2_tail_avg_rank = dict()

    for s in support_2_head_ranks:
        support_2_head_avg_rank[s] = np.average(support_2_head_ranks[s])

    for s in support_2_tail_ranks:
        support_2_tail_avg_rank[s] = np.average(support_2_tail_ranks[s])

    return support_2_head_avg_rank, support_2_tail_avg_rank


dataset_name = FB15K
models_names = [ROTATE]

test_fact_2_support = tfidf_support.read(Dataset(dataset_name))
for model_name in models_names:
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name)

    support_2_head_avg_rank, support_2_tail_avg_rank = process(model_entries, test_fact_2_support)

    plot_dict(support_2_head_avg_rank, model_name + " support vs avg head rank", "support", "avg head rank of test facts with that support")
    plot_dict(support_2_tail_avg_rank, model_name + " support vs avg tail rank", "support", "avg head rank of test facts with that support")
