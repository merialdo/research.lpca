import html
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


dataset_name = YAGO3_10
dataset = Dataset(dataset_name)
entities_count = len(dataset.entities)

all_test_facts_number = float(len(dataset.test_triples))


test_fact_2_peer_classes = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and plot the percentage of facts in each peer class ===
st_peers_class_2_overall_counts = defaultdict(lambda: 0.0)
st_peers_class_2_percentage = defaultdict(lambda: 0.0)

# we are going to use source-target peer classes here.
# for short, we call them ST peer classes

# count the occurrences for each ST peer class and overall
for test_fact in dataset.test_triples:
    test_fact_key = html.unescape(";".join(test_fact))
    peer_class  = test_fact_2_peer_classes[test_fact_key]
    head_peers, tail_peers = peer_class.split("__")

    head_prediction_st_peers_class = tail_peers + "__" + head_peers
    tail_prediction_st_peers_class = head_peers + "__" + tail_peers

    st_peers_class_2_overall_counts[head_prediction_st_peers_class] += 1
    st_peers_class_2_overall_counts[tail_prediction_st_peers_class] += 1

# compute the percentage for each ST peer class
for st_peers_class in st_peers_class_2_overall_counts:
    st_peers_class_2_percentage[st_peers_class] = float(st_peers_class_2_overall_counts[st_peers_class]) / (2*all_test_facts_number)

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

    st_peer_class_2_ranks = defaultdict(lambda: [])

    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=entities_count)

    for entry in model_entries:

        head, relation, tail = entry['head'], entry['relation'], entry['tail']

        peer_class = test_fact_2_peer_class[";".join([head, relation, tail])]
        head_peers, tail_peers = peer_class.split("__")

        head_prediction_st_peers_class = tail_peers + "__" + head_peers
        tail_prediction_st_peers_class = head_peers + "__" + tail_peers

        st_peer_class_2_ranks[head_prediction_st_peers_class].append(entry['head_rank_filtered'])
        st_peer_class_2_ranks[tail_prediction_st_peers_class].append(entry['tail_rank_filtered'])


    st_peer_class_2_mrr = dict()

    for peer_class in PEER_CLASSES:
        if st_peers_class_2_overall_counts[peer_class] > 0:
            st_peer_class_2_mrr[peer_class] = compute_mrr(st_peer_class_2_ranks[peer_class])
        else:
            st_peer_class_2_ranks[peer_class] = None

    mrr_matrix = np.zeros(shape=(len(PEER_INTERVALS), len(PEER_INTERVALS)), dtype=np.float)


    for i in range(len(PEER_INTERVALS)):
        source_peer_interval = PEER_INTERVALS[i]
        source_peer_interval_str = str(source_peer_interval[0]) + "-" + str(source_peer_interval[1])
        for j in range(len(PEER_INTERVALS)):
            target_peer_interval = PEER_INTERVALS[j]
            target_peer_interval_str = str(target_peer_interval[0]) + "-" + str(target_peer_interval[1])

            st_peer_class = source_peer_interval_str + "__" + target_peer_interval_str

            if st_peer_class in st_peer_class_2_mrr:
                mrr_matrix[i, j] = st_peer_class_2_mrr[st_peer_class]
            else:
                mrr_matrix[i, j] = None

    sns.heatmap(mrr_matrix,
                linewidth=0.5,
                annot=True,
                square=True,
                xticklabels=heatmap_x_ticks,
                yticklabels=heatmap_y_ticks,
                cmap="coolwarm_r",
                vmin=0.0,
                vmax=1.0)
    plt.xlabel("Target Peers")
    plt.ylabel("Source Peers")
    plt.title(model_name + " MRR by number of peers")

    plt.show()



facts_in_peer_class_percentages_matrix = np.zeros(shape=(len(PEER_INTERVALS), len(PEER_INTERVALS)), dtype=np.float)

for i in range(len(PEER_INTERVALS)):
    head_peer_interval = PEER_INTERVALS[i]
    head_peer_interval_str = str(head_peer_interval[0]) + "-" + str(head_peer_interval[1])
    for j in range(len(PEER_INTERVALS)):
        tail_peer_interval = PEER_INTERVALS[j]
        tail_peer_interval_str = str(tail_peer_interval[0]) + "-" + str(tail_peer_interval[1])
        peer_class = head_peer_interval_str + "__" + tail_peer_interval_str

        if st_peers_class_2_percentage[peer_class] != 0:
            facts_in_peer_class_percentages_matrix[i, j] = round(st_peers_class_2_percentage[peer_class], 4)
        else:
            facts_in_peer_class_percentages_matrix[i, j] = None

ax = sns.heatmap(facts_in_peer_class_percentages_matrix,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=heatmap_x_ticks,
            yticklabels=heatmap_y_ticks,
            cmap="OrRd",
            vmin=0.0,
            vmax=0.3)
ax.set(xlabel="Source peers interval", ylabel='Target peers interval')
plt.show()

for model_name in models.ALL_MODEL_NAMES :
#for model_name in [models.TRANSE]:
    plot_mrr_heatmap_for(model_name, dataset_name, test_fact_2_peer_classes)