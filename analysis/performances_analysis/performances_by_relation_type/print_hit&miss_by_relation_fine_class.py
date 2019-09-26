import math

import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees, relation_mentions
from dataset_analysis.relation_cardinalities import relation_coarse_classes, relation_fine_classes
from dataset_analysis.relation_cardinalities.relation_fine_classes import FINE_CLASSES
from datasets import FB15K
from io_utils import *
from models import ROTATE

model_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)

rel_2_class = relation_fine_classes.read(FB15K, return_rel_2_class=True)


overall_all = 0.0
fine_class_2_overall_counts = dict()
for fine_class in FINE_CLASSES:
    fine_class_2_overall_counts[fine_class] = 0.0

for entry in model_entries:
    relation = entry['relation']
    fine_class = rel_2_class[relation]
    overall_all += 1
    fine_class_2_overall_counts[fine_class] += 1

print("Overall relation types ratios in test set")
msg = "\t"
for fine_class in fine_class_2_overall_counts:
    overall_percentage = float(fine_class_2_overall_counts[fine_class]) * 100 / float(overall_all)
    msg += fine_class + ": " + str(round(overall_percentage, 2)) + "%; "
print(msg)
print()


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

    relation_type = rel_2_class[relation]

    if head_rank_filtered == 1:
        all_head_hits+=1
        head_hits[relation_type] += 1
    else:
        all_head_misses+=1
        head_misses[relation_type] += 1

    if tail_rank_filtered == 1:
        all_tail_hits+=1
        tail_hits[relation_type] += 1
    else:
        all_tail_misses+=1
        tail_misses[relation_type] += 1


print("All head predictions: " + str(all_head_hits + all_head_misses))
print()
print("All head hits: " + str(all_head_hits))
for fine_class in FINE_CLASSES:
    if fine_class_2_overall_counts[fine_class] != 0:
        perc = float(head_hits[fine_class])*100/fine_class_2_overall_counts[fine_class]
        perc = round(perc, 2)
    else:
        perc = 0.00
    print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")

print()
print("All head misses: " + str(all_head_misses))

for fine_class in FINE_CLASSES:
    if fine_class_2_overall_counts[fine_class] != 0:
        perc = float(head_misses[fine_class])*100/fine_class_2_overall_counts[fine_class]
        perc = round(perc, 2)
    else:
        perc = 0.00

    print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")

print()
print()


print("All tail predictions: " + str(all_tail_hits + all_tail_misses))
print()
print("All tail hits: " + str(all_tail_hits))
for fine_class in FINE_CLASSES:
    perc = float(tail_hits[fine_class])*100/fine_class_2_overall_counts[fine_class]
    perc = round(perc, 2)
    print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")

print()
print("All tail misses: " + str(all_tail_misses))
for fine_class in FINE_CLASSES:
    perc = float(tail_misses[fine_class])*100/fine_class_2_overall_counts[fine_class]
    perc = round(perc, 2)
    print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")
