import math

from dataset_analysis.paths import tfidf_support
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset
from io_utils import *
from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE
from performances import read_filtered_ranks_entries_for
import numpy as np

dataset_name = FB15K
model_entries = read_filtered_ranks_entries_for(ROTATE, dataset_name)

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
    return np.average([1.0/rank for rank in ranks])

test_fact_2_path_support = tfidf_support.read(Dataset(dataset_name))
path_support_bucket_2_test_facts = defaultdict(lambda:[])
for test_fact in test_fact_2_path_support:
    support = test_fact_2_path_support[test_fact]
    support_bucket = get_support_bucket_from_support(support)
    path_support_bucket_2_test_facts[support_bucket].append(test_fact)


all_test_facts_count = float(len(test_fact_2_path_support))
path_support_bucket_2_facts_count = dict()
path_support_bucket_2_facts_percentage = dict()

for path_support_bucket in BUCKETS:
    test_facts_count = float(len(path_support_bucket_2_test_facts[path_support_bucket]))
    path_support_bucket_2_facts_count[path_support_bucket] = test_facts_count

    test_facts_percentage = test_facts_count/all_test_facts_count
    path_support_bucket_2_facts_percentage[path_support_bucket] = test_facts_percentage

# print the percentage of facts in each sibling class
print()
print("Overall relation path support class ratios in test set")
header_row = []
values_row = []
for path_support_bucket in sorted(path_support_bucket_2_test_facts.keys()):
    header_row.append(str(path_support_bucket[0]) + " - " + str(path_support_bucket[1]))
    perc = round(path_support_bucket_2_facts_percentage[path_support_bucket]*100, 2)
    values_row.append((str(perc) + "  "))
print("\t".join(header_row))
print("\t\t".join(values_row))

# === count and print the percentage of hits@1 for each sibling class ===

bucket_2_head_ranks = defaultdict(lambda: [])
bucket_2_tail_ranks = defaultdict(lambda: [])


for entry in model_entries:

    head = entry['head']
    relation = entry['relation']
    tail = entry['tail']
    head_rank_filtered = entry['head_rank_filtered']
    tail_rank_filtered = entry['tail_rank_filtered']

    path_support = test_fact_2_path_support[";".join([head, relation, tail])]
    path_support_bucket = get_support_bucket_from_support(path_support)

    bucket_2_head_ranks[path_support_bucket].append(float(head_rank_filtered))
    bucket_2_tail_ranks[path_support_bucket].append(float(tail_rank_filtered))

print()
print("Head MRR")

values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        mrr = round(compute_mrr(bucket_2_head_ranks[bucket]), 2)
        values_row.append(str(mrr) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))

print()

print("Tail MRR")
values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        mrr = round(compute_mrr(bucket_2_tail_ranks[bucket]), 2)
        values_row.append(str(mrr) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))
