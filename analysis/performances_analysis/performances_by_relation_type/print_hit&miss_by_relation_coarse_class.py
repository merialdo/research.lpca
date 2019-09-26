import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees, relation_mentions
from dataset_analysis.relation_cardinalities import relation_coarse_classes
from datasets import FB15K
from io_utils import *
from models import ROTATE

model_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)

rel_2_class = relation_coarse_classes.read(FB15K, return_rel_2_class=True)

overall_one_2_one_count = 0.0
overall_many_2_one_count = 0.0
overall_one_2_many_count = 0.0
overall_many_2_many_count = 0.0
overall_all = 0.0
for entry in model_entries:
    relation = entry['relation']

    overall_all += 1
    if rel_2_class[relation] == "one to one":
        overall_one_2_one_count += 1
    elif rel_2_class[relation] == "one to many":
        overall_one_2_many_count += 1
    elif rel_2_class[relation] == "many to one":
        overall_many_2_one_count += 1
    else:
        overall_many_2_many_count += 1

overall_one_2_one_percentage = float(overall_one_2_one_count)*100/float(overall_all)
overall_many_2_one_percentage = float(overall_many_2_one_count)*100/float(overall_all)
overall_one_2_many_percentage = float(overall_one_2_many_count)*100/float(overall_all)
overall_many_2_many_percentage = float(overall_many_2_many_count)*100/float(overall_all)
print("Overall relation types ratios in test set")
print("\tOne to one: %f%%; One to many: %f%%; Many to one: %f%%; Many to many: %f%%;" %
      (overall_one_2_one_percentage, overall_many_2_one_percentage, overall_one_2_many_percentage, overall_many_2_many_percentage))
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
print("\tone to one: %f%% of all one to one head questions" % (float(head_hits["one to one"])*100/overall_one_2_one_count))
print("\tone to many: %f%% of all one to many head questions" % (float(head_hits["one to many"])*100/overall_one_2_many_count))
print("\tmany to one: %f%% of all many to one head questions" % (float(head_hits["many to one"])*100/overall_many_2_one_count))
print("\tmany to many: %f%% of all many to many head questions" % (float(head_hits["many to many"])*100/overall_many_2_many_count))
print()
print("All head misses: " + str(all_head_misses))
print("\tone to one: %f%% of all one to one head questions" % (float(head_misses["one to one"])*100/overall_one_2_one_count))
print("\tone to many: %f%% of all one to many head questions" % (float(head_misses["one to many"])*100/overall_one_2_many_count))
print("\tmany to one: %f%% of all many to one head questions" % (float(head_misses["many to one"])*100/overall_many_2_one_count))
print("\tmany to many: %f%% of all many to many head questions" % (float(head_misses["many to many"])*100/overall_many_2_many_count))
print()
print()
print("All tail predictions: " + str(all_tail_hits + all_tail_misses))
print()
print("All tail hits: " + str(all_tail_hits))
print("\tone to one: %f%% of all one to one tail questions" % (float(tail_hits["one to one"])*100/overall_one_2_one_count))
print("\tone to many: %f%% of all one to many tail questions" % (float(tail_hits["one to many"])*100/overall_one_2_many_count))
print("\tmany to one: %f%% of all many to one tail questions" % (float(tail_hits["many to one"])*100/overall_many_2_one_count))
print("\tmany to many: %f%% of all many to many tail questions" % (float(tail_hits["many to many"])*100/overall_many_2_many_count))
print()
print("All tail misses: " + str(all_tail_misses))
print("\tone to one: %f%% of all one to one tail questions" % (float(tail_misses["one to one"])*100/overall_one_2_one_count))
print("\tone to many: %f%% of all one to many tail questions" % (float(tail_misses["one to many"])*100/overall_one_2_many_count))
print("\tmany to one: %f%% of all many to one tail questions" % (float(tail_misses["many to one"])*100/overall_many_2_one_count))
print("\tmany to many: %f%% of all many to many tail questions" % (float(tail_misses["many to many"])*100/overall_many_2_many_count))
