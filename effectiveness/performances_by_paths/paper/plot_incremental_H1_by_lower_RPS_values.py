import copy
import html
import math

from dataset_analysis.paths import tfidf_support
from line_styles import MODEL_LINE_COLOR, MODEL_LINE_STYLE
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

dataset_name = YAGO3_10
dataset = Dataset(dataset_name)
all_models_names = models.ALL_MODEL_NAMES

entities_count = len(dataset.entities)
all_test_facts_number = float(len(dataset.test_triples))

BUCKETS=( (0.0, 0.1),
          (0.1, 0.2),
          (0.2, 0.3),
          (0.3, 0.4),
          (0.4, 0.5),
          (0.5, 0.6),
          (0.6, 0.7),
          (0.7, 0.8),
          (0.8, 0.9),
          (0.9, 1.0) )

def get_support_bucket_from_support(support_value):
    for bucket in BUCKETS:
        if bucket[0] <= support_value < bucket[1]:
            return bucket
    if support_value == 1.0:
        return 0.9, 1.0

# === count and percentage of facts in each RPS interval ===
max1_RPS_bucket_2_count = defaultdict(lambda: 0.0)
max1_RPS_bucket_2_percentage = defaultdict(lambda: 0.0)

# === count and percentage of facts in each RPS interval ===
max2_RPS_bucket_2_count = defaultdict(lambda: 0.0)
max2_RPS_bucket_2_percentage = defaultdict(lambda: 0.0)


test_fact_2_max1_RPS = tfidf_support.read(dataset, 1)
test_fact_2_max2_RPS = tfidf_support.read(dataset, 2)

for fact in dataset.test_triples:
    fact_key = html.unescape(";".join(fact))
    max1_RPS_value = test_fact_2_max1_RPS[fact_key]
    max2_RPS_value = test_fact_2_max2_RPS[fact_key]

    max1_RPS_bucket = get_support_bucket_from_support(max1_RPS_value)
    max2_RPS_bucket = get_support_bucket_from_support(max2_RPS_value)

    max1_RPS_bucket_2_count[max1_RPS_bucket] += 2 # head and tail
    max2_RPS_bucket_2_count[max2_RPS_bucket] += 2 # head and tail

max1_RPS_bucket_2_incremental_count = copy.deepcopy(max1_RPS_bucket_2_count)
max2_RPS_bucket_2_incremental_count = copy.deepcopy(max2_RPS_bucket_2_count)

for i in range(len(BUCKETS)):
    start_bucket = BUCKETS[i]
    for j in range(i + 1, len(BUCKETS)):
        bucket_to_increment = BUCKETS[j]
        max1_RPS_bucket_2_incremental_count[bucket_to_increment] += max1_RPS_bucket_2_count[start_bucket]
        max2_RPS_bucket_2_incremental_count[bucket_to_increment] += max2_RPS_bucket_2_count[start_bucket]

for bucket in BUCKETS:
    max1_RPS_bucket_2_percentage[bucket] = max1_RPS_bucket_2_incremental_count[bucket]/(float(all_test_facts_number)*2) # head and tail
    max2_RPS_bucket_2_percentage[bucket] = max2_RPS_bucket_2_incremental_count[bucket]/(float(all_test_facts_number)*2) # head and tail



def get_bucket_2_incremental_h1(dataset, model_name, fact_2_support, rps_bucket_2_incremental_count):
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset.name, entity_number=len(dataset.entities))

    RPS_bucket_2_h1 = defaultdict(lambda : 0)

    for entry in model_entries:

        head = entry['head']
        relation = entry['relation']
        tail = entry['tail']
        head_rank_filtered = entry['head_rank_filtered']
        tail_rank_filtered = entry['tail_rank_filtered']

        fact_key = html.unescape(";".join([head, relation, tail]))
        RPS_value = fact_2_support[fact_key]
        RPS_bucket = get_support_bucket_from_support(RPS_value)

        if head_rank_filtered == 1:
            RPS_bucket_2_h1[RPS_bucket] += 1

        if tail_rank_filtered == 1:
            RPS_bucket_2_h1[RPS_bucket] += 1

    support_bucket_2_incremental_h1 = copy.deepcopy(RPS_bucket_2_h1)

    for i in range(len(BUCKETS)):
        start_bucket = BUCKETS[i]
        for j in range(i+1, len(BUCKETS)):
            bucket_to_increment = BUCKETS[j]
            support_bucket_2_incremental_h1[bucket_to_increment] += RPS_bucket_2_h1[start_bucket]

    support_bucket_2_incremental_h1_percentage = dict()

    for bucket in BUCKETS:
        support_bucket_2_incremental_h1_percentage[bucket] = support_bucket_2_incremental_h1[bucket]/float(rps_bucket_2_incremental_count[bucket])

    return support_bucket_2_incremental_h1_percentage


x = [str(b) for (a, b) in BUCKETS]

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6, 2)
ax_max1_plots = fig.add_subplot(gs[0:5, 0:1])
ax_max1_dist = fig.add_subplot(gs[5:6, 0:1])
ax_max2_plots = fig.add_subplot(gs[0:5, 1:2])
ax_max2_dist = fig.add_subplot(gs[5:6, 1:2])


for i in range(len(all_models_names)):
    model_name = all_models_names[i]

    max1_RPS_bucket_2_incremental_h1_perc = get_bucket_2_incremental_h1(dataset, model_name, test_fact_2_max1_RPS, max1_RPS_bucket_2_incremental_count)
    max2_RPS_bucket_2_incremental_h1_perc = get_bucket_2_incremental_h1(dataset, model_name, test_fact_2_max2_RPS, max2_RPS_bucket_2_incremental_count)

    max1_y = [max1_RPS_bucket_2_incremental_h1_perc[key] for key in BUCKETS]
    max2_y = [max2_RPS_bucket_2_incremental_h1_perc[key] for key in BUCKETS]

    max1_lines = ax_max1_plots.plot(x, max1_y, label=model_name, linewidth=2)
    max2_lines = ax_max2_plots.plot(x, max2_y, label=model_name, linewidth=2)

    #color_index = i/2*1/8 if i%2 == 0 else (i-1)/2*1/8
    # color = cm(color_index)
    #lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    #print(color_index)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]

    max1_lines[0].set_color(color)
    max1_lines[0].set_alpha(0.8)
    max1_lines[0].set_linestyle(style)

    max2_lines[0].set_color(color)
    max2_lines[0].set_alpha(0.8)
    max2_lines[0].set_linestyle(style)

ax_max1_plots.grid()
ax_max2_plots.grid()
ax_max1_dist.grid()
ax_max2_dist.grid()

#plt.yscale("log")
ax_max1_plots.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=11)
#plt.grid()
#plt.show()

max1_y = [max1_RPS_bucket_2_percentage[key] for key in BUCKETS]
max2_y = [max2_RPS_bucket_2_percentage[key] for key in BUCKETS]

max1_lines = ax_max1_dist.plot(x, max1_y, label="distribution", linewidth=2)
max1_lines[0].set_color("#1123ba")
max2_lines = ax_max2_dist.plot(x, max2_y, label="distribution", linewidth=2)
max2_lines[0].set_color("#1123ba")

base = [0 for key in BUCKETS]

ax_max1_dist.fill_between(x, max1_y, color='#1123ba')
ax_max2_dist.fill_between(x, max2_y, color='#1123ba')


max1_lines = ax_max1_dist.plot(x, max1_y, label="distribution", linewidth=2)
max1_lines[0].set_color("#1123ba")
ax_max1_dist.fill_between(x, max1_y, color='#1123ba')
max2_lines = ax_max2_dist.plot(x, max2_y, label="distribution", linewidth=2)
max2_lines[0].set_color("#1123ba")
ax_max2_dist.fill_between(x, max2_y, color='#1123ba')

ax_max1_plots.set_ylabel("H@1 of facts up to that RPS value", fontsize=12)
ax_max1_dist.set_xlabel("RPS value (max path length = 1)", fontsize=12)

ax_max2_plots.set_ylabel("H@1 of facts up to that RPS value", fontsize=12)
ax_max2_dist.set_xlabel("RPS value (max path length = 2)", fontsize=12)

ax_max1_plots.tick_params(axis="x", labelsize=12)
ax_max1_plots.tick_params(axis="y", labelsize=12)
ax_max1_dist.tick_params(axis="x", labelsize=12)
ax_max1_dist.tick_params(axis="y", labelsize=12)

ax_max2_plots.tick_params(axis="x", labelsize=12)
ax_max2_plots.tick_params(axis="y", labelsize=12)
ax_max2_dist.tick_params(axis="x", labelsize=12)
ax_max2_dist.tick_params(axis="y", labelsize=12)

#plt.subplots_adjust(left=0.05, bottom=0.15, right=0.87, top=0.93, wspace=0.20, hspace=1.0)
plt.subplots_adjust(left=0.05, bottom=0.22, right=0.99, top=0.95, wspace=0.44, hspace=0.99)

#plt.grid()
plt.show()