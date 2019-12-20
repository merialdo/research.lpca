import copy
import html
import math

from line_styles import MODEL_LINE_COLOR, MODEL_LINE_STYLE
from models import *
import models
import performances
from dataset_analysis.peers import peer_classes
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset
from io_utils import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

dataset_name = FB15K

dataset_name_2_y_interval={FB15K: [0, 1],
                           FB15K_237: [0, 0.6],
                           WN18: [0.3, 1],
                           WN18RR: [0, 0.6],
                           YAGO3_10: [0, 0.7]}

dataset = Dataset(dataset_name)
all_models_names = models.ALL_MODEL_NAMES

entities_count = len(dataset.entities)
all_test_facts_number = float(len(dataset.test_triples))

model_2_entries = dict()
for model_name in all_models_names:
    entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=entities_count)
    model_2_entries[model_name] = entries

test_fact_2_peer_class = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and plot the percentage of facts in each peer class ===
source_peers_interval_2_count = defaultdict(lambda: 0.0)
source_peers_interval_2_percentage = defaultdict(lambda: 0.0)
target_peers_interval_2_count = defaultdict(lambda: 0.0)
target_peers_interval_2_percentage = defaultdict(lambda: 0.0)

# count the occurrences for each ST peer class and overall
for test_fact in dataset.test_triples:
    test_fact_key = html.unescape(";".join(test_fact))
    peer_class  = test_fact_2_peer_class[test_fact_key]
    head_peers, tail_peers = peer_class.split("__")

    head_prediction_source_peers = tail_peers
    tail_prediction_source_peers = head_peers
    head_prediction_target_peers = head_peers
    tail_prediction_target_peers = tail_peers

    source_peers_interval_2_count[head_prediction_source_peers] += 1
    source_peers_interval_2_count[tail_prediction_source_peers] += 1
    target_peers_interval_2_count[head_prediction_target_peers] += 1
    target_peers_interval_2_count[tail_prediction_target_peers] += 1

# compute the percentage for each ST peer class
for peers_interval in source_peers_interval_2_count:
    source_peers_interval_2_percentage[peers_interval] = float(source_peers_interval_2_count[peers_interval]) / (2*all_test_facts_number)
    target_peers_interval_2_percentage[peers_interval] = float(target_peers_interval_2_count[peers_interval]) / (2*all_test_facts_number)

keys = ['0-1', '1-2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-128', '128-inf']
source_peers_interval_2_incremental_percentage = source_peers_interval_2_percentage.copy()
for i in range(len(keys)):
    key = keys[i]
    for j in range(i+1, len(keys)):
        cur_key = keys[j]
        source_peers_interval_2_incremental_percentage[cur_key] += source_peers_interval_2_percentage[key]

target_peers_interval_2_incremental_percentage = target_peers_interval_2_percentage.copy()
for i in range(len(keys)):
    key = keys[i]
    for j in range(i+1, len(keys)):
        cur_key = keys[j]
        target_peers_interval_2_incremental_percentage[cur_key] += target_peers_interval_2_percentage[key]

x = [int(key.split("-")[0]) for key in keys]

def get_incremental_h1_by_source_peers_for(model_name):
    entries = model_2_entries[model_name]

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
        source_peers_interval_2_incremental_hits1[keys[i]] = float(hits_1_count)/float(len(ranks))

    return source_peers_interval_2_incremental_hits1




def get_incremental_h1_by_target_peers_for(model_name):
    entries = model_2_entries[model_name]
    target_peers_interval_2_ranks = defaultdict(lambda: [])
    for entry in entries:
        peer_class = test_fact_2_peer_class[entry['head'] + ";" + entry['relation'] + ";" + entry['tail']]
        head_peers_class, tail_peers_class = peer_class.split("__")

        target_peers_interval_2_ranks[head_peers_class].append(entry['head_rank_filtered'])
        target_peers_interval_2_ranks[tail_peers_class].append(entry['tail_rank_filtered'])

    target_peers_interval_2_incremental_ranks = dict()
    for key in keys:
        target_peers_interval_2_incremental_ranks[key] = target_peers_interval_2_ranks[key].copy()

    for i in range(len(keys)):
        key = keys[i]
        for j in range(i + 1, len(keys)):
            cur_key = keys[j]
            target_peers_interval_2_incremental_ranks[cur_key] += target_peers_interval_2_ranks[key]

    target_peers_interval_2_incremental_hits1 = dict()
    for i in range(len(keys)):
        ranks = np.array(target_peers_interval_2_incremental_ranks[keys[i]])
        hits_1_count = np.sum(ranks == 1.0)
        target_peers_interval_2_incremental_hits1[keys[i]] = float(hits_1_count)/float(len(ranks))

    return target_peers_interval_2_incremental_hits1

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6, 2)
ax_source_peers_res = fig.add_subplot(gs[0:5, 0:1])
ax_source_peers_dist = fig.add_subplot(gs[5:6, 0:1])
ax_target_peers_res = fig.add_subplot(gs[0:5, 1:2])
ax_target_peers_dist = fig.add_subplot(gs[5:6, 1:2])

for i in range(len(all_models_names)):
    model_name = all_models_names[i]
    model_source_peers_interval_2_incremental_h1 = get_incremental_h1_by_source_peers_for(model_name)
    y = [model_source_peers_interval_2_incremental_h1[key] for key in keys]
    lines = ax_source_peers_res.plot(x, y, label=model_name, linewidth=2)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]
    lines[0].set_color(color)
    lines[0].set_alpha(0.85)
    lines[0].set_linestyle(style)
ax_source_peers_res.set_xscale('log', basex=10)
ax_source_peers_res.grid()
ax_source_peers_res.set_ylim(dataset_name_2_y_interval[dataset_name])
ax_source_peers_res.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=11)

y = [source_peers_interval_2_incremental_percentage[key] for key in keys]
ax_source_peers_dist.set_xscale('log', basex=10)
lines = ax_source_peers_dist.plot(x, y, label="distribution")
lines[0].set_color("#920D1C")
ax_source_peers_dist.fill_between(x, y, color='#920D1C')
ax_source_peers_dist.set_ylim([0, 1])
ax_source_peers_dist.grid()

for i in range(len(all_models_names)):
    model_name = all_models_names[i]
    model_target_peers_interval_2_incremental_h1 = get_incremental_h1_by_target_peers_for(model_name)
    y = [model_target_peers_interval_2_incremental_h1[key] for key in keys]

    lines = ax_target_peers_res.plot(x, y, label=model_name, linewidth=2)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]
    lines[0].set_color(color)
    lines[0].set_alpha(0.85)
    lines[0].set_linestyle(style)
ax_target_peers_res.set_xscale('log', basex=10)
ax_target_peers_res.set_ylim(dataset_name_2_y_interval[dataset_name])
ax_target_peers_res.grid()
#ax_target_peers_res.legend(bbox_to_anchor=(1.04,1), loc="upper left")


y = [target_peers_interval_2_incremental_percentage[key] for key in keys]
ax_target_peers_dist.set_xscale('log', basex=10)
lines = ax_source_peers_dist.plot(x, y, label="distribution")
lines[0].set_color("#920D1C")
base = [0 for k in keys]
ax_target_peers_dist.fill_between(x, y, color='#920D1C')
ax_target_peers_dist.grid()

ax_source_peers_res.set_ylabel("H@1 up to that number of source peers", fontsize=12)
ax_source_peers_dist.set_xlabel("Number of source peers", fontsize=12)
ax_target_peers_res.set_ylabel("H@1 up to that number of target peers", fontsize=12)
ax_target_peers_dist.set_xlabel("Number of target peers", fontsize=12)

ax_source_peers_res.tick_params(axis="x", labelsize=12)
ax_source_peers_res.tick_params(axis="y", labelsize=12)
ax_source_peers_dist.tick_params(axis="x", labelsize=12)
ax_source_peers_dist.tick_params(axis="y", labelsize=12)
ax_target_peers_res.tick_params(axis="x", labelsize=12)
ax_target_peers_res.tick_params(axis="y", labelsize=12)
ax_target_peers_dist.tick_params(axis="x", labelsize=12)
ax_target_peers_dist.tick_params(axis="y", labelsize=12)

plt.subplots_adjust(left=0.05, bottom=0.22, right=0.99, top=0.95, wspace=0.44, hspace=0.99)
plt.show()

