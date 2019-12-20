import copy
import html
import math

from colors import MODEL_COLOR
from models import *
import models
import performances
from dataset_analysis.peers import peer_classes
from dataset_analysis.peers.peer_classes import PEER_INTERVALS
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset
from io_utils import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

dataset_name = WN18
dataset = Dataset(dataset_name)
all_models_names = models.ALL_MODEL_NAMES if dataset_name != YAGO3_10 else models.YAGO_3_10_MODELS


entities_count = len(dataset.entities)
all_test_facts_number = float(len(dataset.test_triples))

test_fact_2_peer_class = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and plot the percentage of facts in each peer class ===
source_peers_interval_2_count = defaultdict(lambda: 0.0)
source_peers_interval_2_percentage = defaultdict(lambda: 0.0)

# count the occurrences for each ST peer class and overall
for test_fact in dataset.test_triples:
    test_fact_key = html.unescape(";".join(test_fact))
    peer_class  = test_fact_2_peer_class[test_fact_key]
    head_peers, tail_peers = peer_class.split("__")

    head_prediction_source_peers = tail_peers
    tail_prediction_source_peers = head_peers

    source_peers_interval_2_count[head_prediction_source_peers] += 1
    source_peers_interval_2_count[tail_prediction_source_peers] += 1

# compute the percentage for each ST peer class
for source_peers_interval in source_peers_interval_2_count:
    source_peers_interval_2_percentage[source_peers_interval] = float(source_peers_interval_2_count[source_peers_interval]) / (2*all_test_facts_number)


keys = ['0-1', '1-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-inf']
source_peers_interval_2_incremental_count = source_peers_interval_2_count.copy()
source_peers_interval_2_incremental_percentage = source_peers_interval_2_percentage.copy()
for i in range(len(keys)):
    key = keys[i]
    for j in range(i+1, len(keys)):
        cur_key = keys[j]
        source_peers_interval_2_incremental_percentage[cur_key] += source_peers_interval_2_percentage[key]


x = [int(key.split("-")[0]) for key in keys]

def get_incremental_h1_for(model_name, dataset_name):
    entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name)
    source_peers_interval_2_ranks = defaultdict(lambda: [])
    for entry in entries:
        peer_class = test_fact_2_peer_class[entry['head'] + ";" + entry['relation'] + ";" + entry['tail']]
        head_peers_class, tail_peers_class = peer_class.split("__")

        source_peers_interval_2_ranks[tail_peers_class].append(entry['head_rank_filtered'])
        source_peers_interval_2_ranks[head_peers_class].append(entry['tail_rank_filtered'])

    source_peers_interval_2_incremental_ranks = dict()
    for key in keys:
        source_peers_interval_2_incremental_ranks[key] = source_peers_interval_2_ranks[key].copy()

    for i in range(len(keys)):
        key = keys[i]
        for j in range(i + 1, len(keys)):
            cur_key = keys[j]
            source_peers_interval_2_incremental_ranks[cur_key] += source_peers_interval_2_ranks[key]

    source_peers_interval_2_incremental_hits1 = dict()
    for i in range(len(keys)):
        ranks = np.array(source_peers_interval_2_incremental_ranks[keys[i]])
        hits_1_count = np.sum(ranks == 1.0)
        source_peers_interval_2_incremental_hits1[keys[i]] = float(hits_1_count)/float(len(entries)*2)

    return source_peers_interval_2_incremental_hits1



NUM_COLORS = 8
LINE_STYLES = ['solid', 'dashed'] # 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)

cm = plt.get_cmap('gist_rainbow')
fig = plt.figure()
ax = fig.add_subplot(111)


all_models_names = [ANYBURL, COMPLEX, ROTATE, TUCKER, RSN, HOLE, DISTMULT, CROSSE, CONVR, CONVE, TORUSE, ANALOGY, SIMPLE, TRANSE, CONVKB, CAPSE]

ideal_points = [source_peers_interval_2_incremental_percentage[k] for k in keys]
ax.fill_between(x, ideal_points, np.max(ideal_points), color='#666666')
ax.plot(x, ideal_points, label="ideal", color="#666666")

for i in range(len(all_models_names)):
    model_name = all_models_names[i]
    model_source_peers_interval_2_incremental_h1 = get_incremental_h1_for(model_name, dataset_name)
    y = [model_source_peers_interval_2_incremental_h1[key] for key in keys]

    lines = ax.plot(x, y, label=model_name)

    #color_index = i/2*1/8 if i%2 == 0 else (i-1)/2*1/8
    # color = cm(color_index)
    #lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    #print(color_index)
    color = MODEL_COLOR[model_name]

    lines[0].set_color(color)
    lines[0].set_alpha(0.8)
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

ax.set_xscale('log', basex=2)


#plt.yscale("log")
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.grid()
plt.show()