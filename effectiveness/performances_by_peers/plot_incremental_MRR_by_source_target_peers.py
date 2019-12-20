import copy
import html
from line_styles import MODEL_LINE_COLOR, MODEL_LINE_STYLE
import models
import performances
from dataset_analysis.peers import peer_classes
from datasets import FB15K, Dataset
from io_utils import *
import numpy as np
import matplotlib.pylab as plt

dataset_name = FB15K
dataset = Dataset(dataset_name)
all_models_names = models.ALL_MODEL_NAMES
entities_count = len(dataset.entities)
all_test_facts_number = float(len(dataset.test_triples))

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

def compute_mrr(ranks):
    mrr = np.average([1.0/rank for rank in ranks])
    return round(mrr, 2)

def get_interval_2_incremental_mrr(dataset, model_name):
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset.name, entity_number=len(dataset.entities))

    source_peers_interval_2_ranks = defaultdict(lambda : [])
    target_peers_interval_2_ranks = defaultdict(lambda : [])

    for entry in model_entries:

        head, relation, tail, head_rank_filtered, tail_rank_filtered = \
            entry['head'], entry['relation'], entry['tail'], entry['head_rank_filtered'], entry['tail_rank_filtered']

        fact_key = html.unescape(";".join([head, relation, tail]))
        peer_class = test_fact_2_peer_class[fact_key]
        head_peers, tail_peers = peer_class.split("__")

        source_peers_interval_2_ranks[head_peers].append(tail_rank_filtered)
        source_peers_interval_2_ranks[tail_peers].append(head_rank_filtered)
        target_peers_interval_2_ranks[head_peers].append(head_rank_filtered)
        target_peers_interval_2_ranks[tail_peers].append(tail_rank_filtered)

    source_peers_interval_2_incremental_ranks = copy.deepcopy(source_peers_interval_2_ranks)
    target_peers_interval_2_incremental_ranks = copy.deepcopy(target_peers_interval_2_ranks)

    for i in range(len(keys)):
        start_key = keys[i]
        for j in range(i+1, len(keys)):
            key_to_increment = keys[j]
            source_peers_interval_2_incremental_ranks[key_to_increment] += source_peers_interval_2_ranks[start_key]
            target_peers_interval_2_incremental_ranks[key_to_increment] += target_peers_interval_2_ranks[start_key]

    source_peer_class_2_incremental_mrr = dict()
    target_peer_class_2_incremental_mrr = dict()

    for key in keys:
        source_peer_class_2_incremental_mrr[key] = compute_mrr(source_peers_interval_2_incremental_ranks[key])
        target_peer_class_2_incremental_mrr[key] = compute_mrr(target_peers_interval_2_incremental_ranks[key])

    return source_peer_class_2_incremental_mrr, target_peer_class_2_incremental_mrr


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6, 2)
ax_source_peers_res = fig.add_subplot(gs[0:5, 0:1])
ax_source_peers_dist = fig.add_subplot(gs[5:6, 0:1])
ax_target_peers_res = fig.add_subplot(gs[0:5, 1:2])
ax_target_peers_dist = fig.add_subplot(gs[5:6, 1:2])


for i in range(len(all_models_names)):
    model_name = all_models_names[i]
    model_source_peers_interval_2_incremental_mrr, model_target_peers_interval_2_incremental_h1 = get_interval_2_incremental_mrr(dataset, model_name)

    y = [model_source_peers_interval_2_incremental_mrr[key] for key in keys]
    lines = ax_source_peers_res.plot(x, y, label=model_name, linewidth=2)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]
    lines[0].set_color(color)
    lines[0].set_alpha(0.8)
    lines[0].set_linestyle(style)

    y = [model_target_peers_interval_2_incremental_h1[key] for key in keys]
    lines = ax_target_peers_res.plot(x, y, label=model_name, linewidth=2)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]
    lines[0].set_color(color)
    lines[0].set_alpha(0.8)
    lines[0].set_linestyle(style)

ax_source_peers_res.set_xscale('log', basex=2)
ax_source_peers_res.grid()
ax_source_peers_res.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=11)
ax_source_peers_res.set_ylim([0, 1])

ax_target_peers_res.set_xscale('log', basex=2)
ax_target_peers_res.grid()
# ax_target_peers_res.legend(bbox_to_anchor=(1.04,1), loc="upper left")
ax_target_peers_res.set_ylim([0, 1])


y = [source_peers_interval_2_incremental_percentage[key] for key in keys]
ax_source_peers_dist.set_xscale('log', basex=2)
lines = ax_source_peers_dist.plot(x, y, label="distribution")
lines[0].set_color("#920D1C")
ax_source_peers_dist.fill_between(x, y, color='#920D1C')
ax_source_peers_dist.grid()

y = [target_peers_interval_2_incremental_percentage[key] for key in keys]
ax_target_peers_dist.set_xscale('log', basex=2)
lines = ax_source_peers_dist.plot(x, y, label="distribution", linewidth=2)
lines[0].set_color("#920D1C")
base = [0 for k in keys]
ax_target_peers_dist.fill_between(x, y, color='#920D1C')
ax_target_peers_dist.grid()

ax_source_peers_res.set_ylabel("MRR up to that number of source peers", fontsize=12)
ax_source_peers_dist.set_xlabel("Number of source peers", fontsize=12)
ax_target_peers_res.set_ylabel("MRR up to that number of target peers", fontsize=12)
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

