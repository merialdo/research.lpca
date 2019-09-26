import math

import models
from dataset_analysis.relation_path_support import tfidf_support
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset, ALL_DATASET_NAMES
from io_utils import *
from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE, COMPLEX
from performances import read_filtered_ranks_entries_for
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# Activating tex in all labels globally
plt.rc('text', usetex=True)
# Adjust font specs as desired (here: closest similarity to seaborn standard)
plt.rc('font', **{'size': 12.0})
plt.rc('text.latex', preamble=r'\usepackage{lmodern}')


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

def get_hits_by_bucket(dataset_name,
                       model_name,
                       fact_2_support,
                       support_bucket_2_facts_count):

    model_entries = read_filtered_ranks_entries_for(model_name, dataset_name)

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

        path_support = fact_2_support[";".join([head, relation, tail])]
        path_support_bucket = get_support_bucket_from_support(path_support)

        if head_rank_filtered == 1:
            all_head_hits += 1
            head_hits[path_support_bucket] += 1
        else:
            all_head_misses += 1
            head_misses[path_support_bucket] += 1

        if tail_rank_filtered == 1:
            all_tail_hits += 1
            tail_hits[path_support_bucket] += 1
        else:
            all_tail_misses += 1
            tail_misses[path_support_bucket] += 1

    head_hits_row = []
    for bucket in BUCKETS:
        if support_bucket_2_facts_count[bucket] > 0:
            hits_1_perc = round(float(head_hits[bucket]) / float(support_bucket_2_facts_count[bucket]), 2)
            head_hits_row.append(hits_1_perc)
        else:
            head_hits_row.append('--')

    tail_hits_row = []
    for bucket in BUCKETS:
        if support_bucket_2_facts_count[bucket] > 0:
            hits_1_perc = round(float(tail_hits[bucket]) / float(support_bucket_2_facts_count[bucket]), 2)
            tail_hits_row.append(str(hits_1_perc) + "   ")
        else:
            values_row.append('--')

    return head_hits_row, tail_hits_row


dataset_name = FB15K_237

test_fact_2_tfidf_path_support = tfidf_support.read(Dataset(dataset_name))
tfidf_path_support_bucket_2_test_facts = defaultdict(lambda: [])
tfidf_path_support_bucket_2_test_facts_count = dict()
tfidf_path_support_bucket_2_test_facts_percentage = dict()


for test_fact in test_fact_2_tfidf_path_support:
    support = test_fact_2_tfidf_path_support[test_fact]
    support_bucket = get_support_bucket_from_support(support)
    tfidf_path_support_bucket_2_test_facts[support_bucket].append(test_fact)

all_test_facts_count = float(len(test_fact_2_tfidf_path_support))

for path_support_bucket in BUCKETS:
    test_facts_count = float(len(tfidf_path_support_bucket_2_test_facts[path_support_bucket]))
    tfidf_path_support_bucket_2_test_facts_count[path_support_bucket] = test_facts_count
    test_facts_percentage = test_facts_count / all_test_facts_count
    tfidf_path_support_bucket_2_test_facts_percentage[path_support_bucket] = test_facts_percentage

model_name_2_head_hits = dict()
model_name_2_tail_hits = dict()

# print the percentage of facts in each sibling class
print()
header_row = []
values_row = []
for bucket in BUCKETS:
    header_row.append(str(bucket[0]) + " - " + str(bucket[1]))
    perc = round(tfidf_path_support_bucket_2_test_facts_percentage[bucket] * 100, 2)
    values_row.append((str(perc) + "  "))
print("\t".join(header_row))
print("\t\t".join(values_row))

models_for_dataset = models.get_models_supporting_dataset(dataset_name)


all_head_hits = np.zeros((len(models_for_dataset), len(BUCKETS)))
all_tail_hits = np.zeros((len(models_for_dataset), len(BUCKETS)))

for i in range(len(models_for_dataset)):
    model = models_for_dataset[i]
    model_head_hits, model_tail_hits = get_hits_by_bucket(dataset_name,
                                                          model,
                                                          test_fact_2_tfidf_path_support,
                                                          tfidf_path_support_bucket_2_test_facts_count)
    all_head_hits[i][:] = model_head_hits
    all_tail_hits[i][:] = model_tail_hits


max_in_head_hits = np.max(all_head_hits, axis=0)

sns.heatmap(all_head_hits,
            mask=all_head_hits == max_in_head_hits,
            linewidth=0.5,
            annot=True,
            xticklabels=header_row,
            yticklabels=models_for_dataset,
            cmap="coolwarm_r")



sns.heatmap(all_head_hits,
            mask=all_head_hits!=max_in_head_hits,
            linewidth=0.5,
            xticklabels=header_row,
            yticklabels=models_for_dataset,
            annot=np.array([r'\underline{' + str(data) + '}'
                            for data in all_head_hits.ravel()]).reshape(np.shape(all_head_hits)),
            # fmt key must be empty, formatting error otherwise
            fmt='',
            cbar=False,
            cmap="coolwarm_r")

plt.show()


max_in_tail_hits = np.max(all_tail_hits, axis=0)

sns.heatmap(all_tail_hits,
            mask=all_tail_hits == max_in_tail_hits,
            linewidth=0.5,
            annot=True,
            xticklabels=header_row,
            yticklabels=models_for_dataset,
            cmap="coolwarm_r")

sns.heatmap(all_tail_hits,
            mask=all_tail_hits != max_in_tail_hits,
            linewidth=0.5,
            xticklabels=header_row,
            yticklabels=models_for_dataset,
            annot=np.array([r'\underline{' + str(data) + '}'
                            for data in all_tail_hits.ravel()]).reshape(np.shape(all_tail_hits)),
            # fmt key must be empty, formatting error otherwise
            fmt='',
            cbar=False,
            cmap="coolwarm_r")

plt.show()
