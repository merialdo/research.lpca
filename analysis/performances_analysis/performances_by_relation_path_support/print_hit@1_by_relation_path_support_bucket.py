import math

from dataset_analysis.relation_path_support import tfidf_support
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset
from io_utils import *
from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE, COMPLEX
from performances import read_filtered_ranks_entries_for


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

dataset_name = FB15K
model_entries = read_filtered_ranks_entries_for(ANYBURL, dataset_name)

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

    path_support = test_fact_2_path_support[";".join([head, relation, tail])]
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

print()
print()
print("All head predictions: " + str(all_head_hits + all_head_misses))
print()

print("All head hits: " + str(all_head_hits))
values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        hits_1_count = head_hits[bucket]
        hits_1_perc = round(float(hits_1_count)/float(path_support_bucket_2_facts_count[bucket]), 2)
        values_row.append(str(hits_1_perc) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))

print()

print("All head misses: " + str(all_head_misses))
values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        misses_1_count = head_misses[bucket]
        misses_1_perc = round(float(misses_1_count)/float(path_support_bucket_2_facts_count[bucket]), 2)
        values_row.append(str(misses_1_perc) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))
print()
print()

print("All tail predictions: " + str(all_tail_hits + all_tail_misses))
print()

print("All tail hits: " + str(all_tail_hits))
values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        hits_1_count = tail_hits[bucket]
        hits_1_perc = round(float(hits_1_count)/float(path_support_bucket_2_facts_count[bucket]), 2)
        values_row.append(str(hits_1_perc) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))

print()
print("All tail misses: " + str(all_tail_misses))
values_row = []
for bucket in BUCKETS:
    if path_support_bucket_2_facts_count[bucket] > 0:
        misses_1_count = tail_misses[bucket]
        misses_1_perc = round(float(misses_1_count)/float(path_support_bucket_2_facts_count[bucket]), 2)
        values_row.append(str(misses_1_perc) + "   ")
    else:
        values_row.append('--')
print("\t".join(header_row))
print("\t\t".join(values_row))
