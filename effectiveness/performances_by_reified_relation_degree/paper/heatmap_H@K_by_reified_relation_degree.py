import math

import models
from dataset_analysis.reified_relation_degree import reified_relation_degree
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset, ALL_DATASET_NAMES
from io_utils import *
# from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE, COMPLEX
from performances import read_filtered_ranks_entries_for
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

dataset_name = FB15K

dataset = Dataset(dataset_name)

K = 1

# Activating tex in all labels globally
plt.rc('text', usetex=True)
# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 12.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(18, 1)
ax1 = fig.add_subplot(gs[0:16, :])
ax2 = fig.add_subplot(gs[16, :])

BUCKETS=( (1, 2),
          (2, 4),
          (4, 8),
          (8, 16),
          (16, 32),
          (32, "inf") )

def get_bucket_from_arity(arity_value):
    if arity_value >= 32:
        return 32, "inf"
    else:
        for bucket in BUCKETS:
            if bucket[0] <= arity_value < bucket[1]:
                return bucket

def get_hits_at_k_by_bucket(k,
                       dataset,
                       model_name,
                       fact_2_arity,
                       arity_bucket_2_facts_count):

    model_entries = read_filtered_ranks_entries_for(model_name, dataset.name, entity_number=len(dataset.entities))

    # === count and print the percentage of hits@1 for each sibling class ===

    all_head_hits = 0
    head_hits = defaultdict(lambda: 0)
    all_head_misses = 0
    head_misses = defaultdict(lambda: 0)

    all_tail_hits = 0
    tail_hits = defaultdict(lambda: 0)
    all_tail_misses = 0
    tail_misses = defaultdict(lambda: 0)

    for entry in model_entries:

        head = entry['head']
        relation = entry['relation']
        tail = entry['tail']
        head_rank_filtered = entry['head_rank_filtered']
        tail_rank_filtered = entry['tail_rank_filtered']

        fact_arity = fact_2_arity[";".join([head, relation, tail])]
        fact_arity_bucket = get_bucket_from_arity(fact_arity)

        if head_rank_filtered <= k:
            all_head_hits += 1
            head_hits[fact_arity_bucket] += 1
        else:
            all_head_misses += 1
            head_misses[fact_arity_bucket] += 1

        if tail_rank_filtered <=k:
            all_tail_hits += 1
            tail_hits[fact_arity_bucket] += 1
        else:
            all_tail_misses += 1
            tail_misses[fact_arity_bucket] += 1

    head_hits_row = []
    for bucket in BUCKETS:
        if arity_bucket_2_facts_count[bucket] > 0:
            hits_1_perc = round(float(head_hits[bucket]) / float(arity_bucket_2_facts_count[bucket]), 2)
            head_hits_row.append(hits_1_perc)
        else:
            head_hits_row.append('--')

    tail_hits_row = []
    for bucket in BUCKETS:
        if arity_bucket_2_facts_count[bucket] > 0:
            hits_1_perc = round(float(tail_hits[bucket]) / float(arity_bucket_2_facts_count[bucket]), 2)
            tail_hits_row.append(str(hits_1_perc) + "   ")
        else:
            tail_hits_row.append('--')

    return head_hits_row, tail_hits_row

# =========================================================================================================

test_fact_2_relation_arity = reified_relation_degree.read(dataset_name, return_fact_2_arity=True)
relation_arity_bucket_2_test_facts = defaultdict(lambda: [])
relation_arity_bucket_2_test_facts_count = dict()
relation_arity_bucket_2_test_facts_percentage = dict()


for test_fact in test_fact_2_relation_arity:
    arity = test_fact_2_relation_arity[test_fact]
    arity_bucket = get_bucket_from_arity(arity)
    relation_arity_bucket_2_test_facts[arity_bucket].append(test_fact)

all_test_facts_count = float(len(test_fact_2_relation_arity))

for arity_bucket in BUCKETS:
    test_facts_count = float(len(relation_arity_bucket_2_test_facts[arity_bucket]))
    relation_arity_bucket_2_test_facts_count[arity_bucket] = test_facts_count
    test_facts_percentage = test_facts_count / all_test_facts_count
    relation_arity_bucket_2_test_facts_percentage[arity_bucket] = test_facts_percentage

model_name_2_head_hits = dict()
model_name_2_tail_hits = dict()

header_row = []
values = np.zeros((1, len(BUCKETS)))

for i in range(len(BUCKETS)):
    header_row.append(str(BUCKETS[i][0]) + " - " + str(BUCKETS[i][1]))
    perc = round(relation_arity_bucket_2_test_facts_percentage[BUCKETS[i]], 2)
    values[0][i] = perc

sns.heatmap(values,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=header_row,
            yticklabels=np.array(["percentage"]),
            cmap="OrRd",
            ax=ax2)

all_head_hits = np.zeros((len(models.ALL_MODEL_NAMES), len(BUCKETS)))
all_tail_hits = np.zeros((len(models.ALL_MODEL_NAMES), len(BUCKETS)))

for i in range(len(models.ALL_MODEL_NAMES)):
    model_name = models.ALL_MODEL_NAMES[i]
    model_head_hits, model_tail_hits = get_hits_at_k_by_bucket(K, dataset, model_name,
                                                                   test_fact_2_relation_arity,
                                                                   relation_arity_bucket_2_test_facts_count)
    all_head_hits[i][:] = model_head_hits
    all_tail_hits[i][:] = model_tail_hits

all_hits = np.round((all_tail_hits + all_head_hits)/2, 2)
max_in_all_hits = np.max(all_hits, axis=0)

print(all_hits)
sns.heatmap(all_hits,
            mask=all_hits == max_in_all_hits,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=header_row,
            yticklabels=models.ALL_MODEL_NAMES,
            cmap="coolwarm_r", ax=ax1)

sns.heatmap(all_hits,
            mask=all_hits != max_in_all_hits,
            linewidth=0.5,
            square=True,
            xticklabels=header_row,
            yticklabels=models.ALL_MODEL_NAMES,
            annot=np.array([r'\underline{' + str(data) + '}'
                            for data in all_hits.ravel()]).reshape(np.shape(all_hits)),
            # fmt key must be empty, formatting error otherwise
            fmt='',
            cbar=False,
            cmap="coolwarm_r", ax=ax1)

plt.show()
