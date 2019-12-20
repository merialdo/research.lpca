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

dataset_name = FB15K
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


def compute_mrr(ranks):
    mrr = np.average([1.0/rank for rank in ranks])
    return round(mrr, 2)

def get_bucket_2_incremental_mrr(dataset, model_name, fact_2_support):
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset.name, entity_number=len(dataset.entities))

    support_bucket_2_head_ranks = defaultdict(lambda : [])
    support_bucket_2_tail_ranks = defaultdict(lambda : [])

    for entry in model_entries:

        head = entry['head']
        relation = entry['relation']
        tail = entry['tail']
        head_rank_filtered = entry['head_rank_filtered']
        tail_rank_filtered = entry['tail_rank_filtered']

        fact_key = html.unescape(";".join([head, relation, tail]))
        support_value = fact_2_support[fact_key]
        support_bucket = get_support_bucket_from_support(support_value)

        support_bucket_2_head_ranks[support_bucket].append(head_rank_filtered)
        support_bucket_2_tail_ranks[support_bucket].append(tail_rank_filtered)

    support_bucket_2_incremental_head_ranks = copy.deepcopy(support_bucket_2_head_ranks)
    support_bucket_2_incremental_tail_ranks = copy.deepcopy(support_bucket_2_tail_ranks)

    for i in range(len(BUCKETS)):
        start_bucket = BUCKETS[i]
        for j in range(i+1, len(BUCKETS)):
            bucket_to_increment = BUCKETS[j]
            support_bucket_2_incremental_head_ranks[bucket_to_increment] += support_bucket_2_head_ranks[start_bucket]
            support_bucket_2_incremental_tail_ranks[bucket_to_increment] += support_bucket_2_tail_ranks[start_bucket]

    support_bucket_2_incremental_head_mrr = dict()
    support_bucket_2_incremental_tail_mrr = dict()
    support_bucket_2_incremental_mrr = dict()

    for bucket in BUCKETS:
        support_bucket_2_incremental_head_mrr[bucket] = compute_mrr(support_bucket_2_incremental_head_ranks[bucket])
        support_bucket_2_incremental_tail_mrr[bucket] = compute_mrr(support_bucket_2_incremental_tail_ranks[bucket])
        support_bucket_2_incremental_mrr[bucket] = (support_bucket_2_incremental_head_mrr[bucket] + support_bucket_2_incremental_tail_mrr[bucket])/2
    return support_bucket_2_incremental_head_mrr, support_bucket_2_incremental_head_mrr, support_bucket_2_incremental_mrr


# === count and percentage of facts in each path support interval ===
path_support_bucket_2_count = defaultdict(lambda: 0.0)
path_support_interval_2_percentage = defaultdict(lambda: 0.0)

test_fact_2_tfidf_path_support = tfidf_support.read(dataset)
for fact in dataset.test_triples:
    fact_key = html.unescape(";".join(fact))
    support_value = test_fact_2_tfidf_path_support[fact_key]
    support_bucket = get_support_bucket_from_support(support_value)

    path_support_bucket_2_count[support_bucket] += 1

support_bucket_2_incremental_count = copy.deepcopy(path_support_bucket_2_count)

for i in range(len(BUCKETS)):
    start_bucket = BUCKETS[i]
    for j in range(i + 1, len(BUCKETS)):
        bucket_to_increment = BUCKETS[j]
        support_bucket_2_incremental_count[bucket_to_increment] += path_support_bucket_2_count[start_bucket]

for bucket in BUCKETS:
    path_support_interval_2_percentage[bucket] = support_bucket_2_incremental_count[bucket]/float(all_test_facts_number)


x = [str(b) for (a, b) in BUCKETS]

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6, 1)
ax1 = fig.add_subplot(gs[0:5, :])
ax2 = fig.add_subplot(gs[5, :])

for i in range(len(all_models_names)):
    model_name = all_models_names[i]
    bucket_2_incremental_head_mrr, bucket_2_incremental_tail_mrr, bucket_2_incremental_mrr = get_bucket_2_incremental_mrr(dataset, model_name, test_fact_2_tfidf_path_support)
    y = [bucket_2_incremental_tail_mrr[key] for key in BUCKETS]

    lines = ax1.plot(x, y, label=model_name, linewidth=2)

    #color_index = i/2*1/8 if i%2 == 0 else (i-1)/2*1/8
    # color = cm(color_index)
    #lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    #print(color_index)
    color = MODEL_LINE_COLOR[model_name]
    style = MODEL_LINE_STYLE[model_name]

    lines[0].set_color(color)
    lines[0].set_alpha(0.8)
    lines[0].set_linestyle(style)

ax1.grid()

#plt.yscale("log")
ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=11)
#plt.grid()
#plt.show()

y = [path_support_interval_2_percentage[key] for key in BUCKETS]
lines = ax2.plot(x, y, label="distribution", linewidth=2)
lines[0].set_color("#1123ba")
base = [0 for key in BUCKETS]
ax2.fill_between(x, y, color='#1123ba')
ax2.grid()

ax1.set_ylabel("MRR of facts up to that RPS value", fontsize=12)
ax2.set_xlabel("RPS value", fontsize=12)

ax1.tick_params(axis="x", labelsize=12)
ax1.tick_params(axis="y", labelsize=12)
ax2.tick_params(axis="x", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)

plt.subplots_adjust(left=0.05, bottom=0.15, right=0.87, top=0.93, wspace=0.20, hspace=1.0)

#plt.grid()
plt.show()

