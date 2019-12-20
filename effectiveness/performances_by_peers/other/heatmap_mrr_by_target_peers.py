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
all_models_names = models.ALL_MODEL_NAMES if dataset_name != YAGO3_10 else models.YAGO_3_10_MODELS
all_models_names.remove(models.STRANSE)
all_predictions_number = float(len(dataset.test_triples))*2

test_fact_2_peer_class = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and plot the percentage of facts in each peer class ===
target_peers_interval_2_count = defaultdict(lambda: 0.0)
target_peers_interval_2_percentage = defaultdict(lambda: 0.0)

# count the occurrences for each ST peer class and overall
for test_fact in dataset.test_triples:
    test_fact_key = html.unescape(";".join(test_fact))
    head_tail_peer_class  = test_fact_2_peer_class[test_fact_key]
    head_peers, tail_peers = head_tail_peer_class.split("__")

    head_prediction_target_peers = head_peers
    tail_prediction_target_peers = tail_peers

    target_peers_interval_2_count[head_prediction_target_peers] += 1
    target_peers_interval_2_count[tail_prediction_target_peers] += 1

# compute the percentage for each ST peer class
for target_peers_interval in target_peers_interval_2_count:
    target_peers_interval_2_percentage[target_peers_interval] = float(target_peers_interval_2_count[target_peers_interval]) / all_predictions_number

# compute the x and y ticks that we will use in all our heatmaps
heatmap_x_ticks = []
heatmap_y_ticks = []

for target_peers_interval in PEER_INTERVALS[:-1]:
    tick = str(target_peers_interval[0]) + " - " + str(target_peers_interval[1])
    heatmap_x_ticks.append(tick)
heatmap_x_ticks.append("> 128")

for mod in all_models_names:
    heatmap_y_ticks.append(mod)

percentages_matrix = np.zeros(shape=(1, len(PEER_INTERVALS)), dtype=np.float)

for j in range(len(PEER_INTERVALS)):
    target_peers_interval = PEER_INTERVALS[j]
    target_peers_interval_str = str(target_peers_interval[0]) + "-" + str(target_peers_interval[1])

    if target_peers_interval_2_percentage[target_peers_interval_str] != 0:
        percentages_matrix[0,j] = round(target_peers_interval_2_percentage[target_peers_interval_str], 4)
    else:
        percentages_matrix[0,j] = None

ax = sns.heatmap(percentages_matrix,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=heatmap_x_ticks,
            yticklabels=[],
            cmap="OrRd",
            vmin=0.0,
            vmax=0.3)
ax.set(xlabel="Peers interval")
plt.show()

# ----------------------------------------------------------------------------------------------------

mrr_matrix = np.zeros(shape=[len(all_models_names), len(PEER_INTERVALS)], dtype=np.float)

def compute_mrr(ranks):
    mrr = np.average([1.0 / rank for rank in ranks])
    return round(mrr, 2)


for i in range(len(all_models_names)):
    model_name = all_models_names[i]

    # === count the MRR for each source peers interval ===

    target_peer_interval_2_ranks_list = defaultdict(lambda: [])

    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=entities_count)

    for entry in model_entries:
        head, relation, tail = entry['head'], entry['relation'], entry['tail']
        peer_class = test_fact_2_peer_class[";".join([head, relation, tail])]
        head_peers_interval, tail_peers_interval = peer_class.split("__")

        head_pred_target_peers_interval = head_peers_interval
        tail_pred_target_peers_interval = tail_peers_interval

        target_peer_interval_2_ranks_list[head_pred_target_peers_interval].append(entry['head_rank_filtered'])
        target_peer_interval_2_ranks_list[tail_pred_target_peers_interval].append(entry['tail_rank_filtered'])


    for j in range(len(PEER_INTERVALS)):
        target_peers_interval = str(PEER_INTERVALS[j][0]) + "-" + str(PEER_INTERVALS[j][1])
        if target_peers_interval_2_count[target_peers_interval] > 0:
            mrr_matrix[i, j] = compute_mrr(target_peer_interval_2_ranks_list[target_peers_interval])
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
ax.set(xlabel="Target peers interval")
plt.show()
