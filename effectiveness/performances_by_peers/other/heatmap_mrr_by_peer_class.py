import math

import models
import performances
from dataset_analysis.peers import peer_classes
from dataset_analysis.peers.peer_classes import PEER_CLASSES, PEER_INTERVALS
from dataset_analysis.paths import tfidf_support
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset, ALL_DATASET_NAMES
from io_utils import *
from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE, COMPLEX
from performances import read_filtered_ranks_entries_for
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


dataset_name = FB15K_237
dataset = Dataset(dataset_name)
test_fact_2_peer_class = peer_classes.read(dataset_name, return_fact_2_class=True)
# === count and print the percentage of facts in each peer class ===
peers_class_2_overall_counts = dict()
peers_class_2_percentage = dict()

# initialize the data structure
for peer_class in PEER_CLASSES:
    peers_class_2_overall_counts[peer_class] = 0.0

overall_all = float(len(dataset.test_triples))

# count the occurrences for each peer class and overall
for test_fact in dataset.test_triples:
    peer_class = test_fact_2_peer_class[";".join(test_fact)]
    peers_class_2_overall_counts[peer_class] += 1

# compute the percentage for each peer class
for peer_class in peers_class_2_overall_counts:
    peers_class_2_percentage[peer_class] = float(peers_class_2_overall_counts[peer_class]) / overall_all


# compute the x and y ticks that we will use in all our heatmaps
heatmap_x_ticks = []
heatmap_y_ticks = []

for i in range(len(PEER_INTERVALS[:-1])):
    tick = str(PEER_INTERVALS[i][0]) + " - " + str(PEER_INTERVALS[i][1])
    heatmap_x_ticks.append(tick)
    heatmap_y_ticks.append(tick)
heatmap_x_ticks.append("> 128")
heatmap_y_ticks.append("> 128")


def compute_mrr(ranks):
    mrr = np.average([1.0/rank for rank in ranks])
    return round(mrr, 2)

def plot_mrr_heatmap_for(model_name, dataset_name, test_fact_2_peer_class):
    # === count the percentage of hits@1 for each peer class ===

    peer_class_2_head_ranks = defaultdict(lambda: [])
    peer_class_2_tail_ranks = defaultdict(lambda: [])

    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name)

    for entry in model_entries:

        head, relation, tail = entry['head'], entry['relation'], entry['tail']

        peer_class = test_fact_2_peer_class[";".join([head, relation, tail])]

        peer_class_2_head_ranks[peer_class].append(entry['head_rank_filtered'])
        peer_class_2_tail_ranks[peer_class].append(entry['tail_rank_filtered'])


    peer_class_2_head_mrr = dict()
    peer_class_2_tail_mrr = dict()

    for peer_class in PEER_CLASSES:
        if peers_class_2_overall_counts[peer_class] > 0:

            peer_class_2_head_mrr[peer_class] = compute_mrr(peer_class_2_head_ranks[peer_class])
            peer_class_2_tail_mrr[peer_class] = compute_mrr(peer_class_2_tail_ranks[peer_class])
        else:
            peer_class_2_head_mrr[peer_class] = None
            peer_class_2_tail_mrr[peer_class] = None

    head_mrr_matrix = np.zeros(shape=(len(PEER_INTERVALS), len(PEER_INTERVALS)), dtype=np.float)
    tail_mrr_matrix = np.zeros(shape=(len(PEER_INTERVALS), len(PEER_INTERVALS)), dtype=np.float)

    for i in range(len(PEER_INTERVALS)):
        head_peer_interval = PEER_INTERVALS[i]
        head_peer_interval_str = str(head_peer_interval[0]) + "-" + str(head_peer_interval[1])
        for j in range(len(PEER_INTERVALS)):
            tail_peer_interval = PEER_INTERVALS[j]
            tail_peer_interval_str = str(tail_peer_interval[0]) + "-" + str(tail_peer_interval[1])
            peer_class = head_peer_interval_str + "__" + tail_peer_interval_str
            head_mrr_matrix[i, j] = peer_class_2_head_mrr[peer_class]
            tail_mrr_matrix[i, j] = peer_class_2_tail_mrr[peer_class]

    sns.heatmap(head_mrr_matrix,
                linewidth=0.5,
                annot=True,
                square=True,
                xticklabels=heatmap_x_ticks,
                yticklabels=heatmap_y_ticks,
                cmap="coolwarm_r",
                vmin=0.0,
                vmax=1.0)
    plt.xlabel("Tail Peers")
    plt.ylabel("Head Peers")
    plt.title(model_name + ": head prediction MRR")
    plt.show()


    sns.heatmap(tail_mrr_matrix,
                linewidth=0.5,
                annot=True,
                square=True,
                xticklabels=heatmap_x_ticks,
                yticklabels=heatmap_y_ticks,
                cmap="coolwarm_r",
                vmin=0.0,
                vmax=1.0)
    plt.xlabel("Tail Peers")
    plt.ylabel("Head Peers")
    plt.title(model_name + ": tail prediction MRR")

    plt.show()



facts_in_peer_class_percentages_matrix = np.zeros(shape=(len(PEER_INTERVALS), len(PEER_INTERVALS)), dtype=np.float)

for i in range(len(PEER_INTERVALS)):
    head_peer_interval = PEER_INTERVALS[i]
    head_peer_interval_str = str(head_peer_interval[0]) + "-" + str(head_peer_interval[1])
    for j in range(len(PEER_INTERVALS)):
        tail_peer_interval = PEER_INTERVALS[j]
        tail_peer_interval_str = str(tail_peer_interval[0]) + "-" + str(tail_peer_interval[1])
        peer_class = head_peer_interval_str + "__" + tail_peer_interval_str

        if peers_class_2_percentage[peer_class] != 0:
            facts_in_peer_class_percentages_matrix[i, j] = round(peers_class_2_percentage[peer_class], 4)
        else:
            facts_in_peer_class_percentages_matrix[i, j] = None

sns.heatmap(facts_in_peer_class_percentages_matrix,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=heatmap_x_ticks,
            yticklabels=heatmap_y_ticks,
            cmap="OrRd")
plt.show()

for model_name in models.ALL_MODEL_NAMES:
    plot_mrr_heatmap_for(model_name, dataset_name, test_fact_2_peer_class)