import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

import datasets
import models
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.peers.peer_classes import PEER_CLASSES, PEER_INTERVALS
from dataset_analysis.peers import peer_classes
from io_utils import *
from performances import read_filtered_ranks_entries_for


dataset_name = datasets.WN18RR
model_entries = read_filtered_ranks_entries_for(models.CONVR, dataset_name)

test_fact_2_class = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and print the percentage of facts in each peer class ===
overall_all = 0.0
peers_class_2_overall_counts = dict()
peers_class_2_percentage = dict()
# initialize the data structure
for fine_class in PEER_CLASSES:
    peers_class_2_overall_counts[fine_class] = 0.0
# count the occurrences for each peer class and overall
for entry in model_entries:
    head, relation, tail = entry['head'], entry['relation'], entry['tail']
    peer_class = test_fact_2_class[";".join([head, relation, tail])]
    overall_all += 1
    peers_class_2_overall_counts[peer_class] += 1
# compute the percentage for each peer class
for peer_class in peers_class_2_overall_counts:
    peers_class_2_percentage[peer_class] = float(peers_class_2_overall_counts[peer_class]) / float(overall_all)

# print the percentage of facts in each peer class
print("Overall peer class ratios in test set")
rows = [";".join([''] + [str(tail_sib_interval[0]) + " <= tail peers < " + str(tail_sib_interval[1]) for tail_sib_interval in PEER_INTERVALS])]
for head_sib_interval in PEER_INTERVALS:
    row_bits = [str(head_sib_interval[0]) + " <= head peers < " + str(head_sib_interval[1])]
    for tail_sib_interval in PEER_INTERVALS:
        peer_class = str(head_sib_interval[0]) + "-" + str(head_sib_interval[1]) + "__" + str(tail_sib_interval[0]) + "-" + str(tail_sib_interval[1])
        row_bits.append(str(round(100*peers_class_2_percentage[peer_class], 2)))
    rows.append(";".join(row_bits).replace(".", ","))
for row in rows:
    print(row)

peers_class_2_head_ranks = defaultdict(lambda:[])
peers_class_2_tail_ranks = defaultdict(lambda:[])

for entry in model_entries:

    head = entry['head']
    relation = entry['relation']
    tail = entry['tail']
    head_rank_filtered = entry['head_rank_filtered']
    tail_rank_filtered = entry['tail_rank_filtered']
    peers_class = test_fact_2_class[";".join([head, relation, tail])]

    peers_class_2_head_ranks[peers_class].append(float(head_rank_filtered))
    peers_class_2_tail_ranks[peers_class].append(float(tail_rank_filtered))


peer_class_2_head_mrr = defaultdict(lambda: 0)
peer_class_2_tail_mrr = defaultdict(lambda: 0)

print("MRR in head predictions: ")
for peers_class in PEER_CLASSES:
    if peers_class_2_overall_counts[peers_class] > 0:
        head_ranks = peers_class_2_head_ranks[peers_class]
        inverse_ranks = [1.0/x for x in head_ranks]
        mrr = numpy.average(inverse_ranks)
        mrr = round(mrr, 2)
        print("\t" + peers_class + ": " + str(mrr))
        peer_class_2_head_mrr[peers_class] = mrr
    else:
        print("\t" + peers_class + ": UNDEFINED")
        peer_class_2_head_mrr[peers_class] = "--"

print()

for peers_class in PEER_CLASSES:
    if peers_class_2_overall_counts[peers_class] > 0:
        tail_ranks = peers_class_2_tail_ranks[peers_class]
        inverse_ranks = [1.0 / x for x in tail_ranks]
        mrr = numpy.average(inverse_ranks)
        mrr = round(mrr, 2)
        print("\t" + peers_class + ": " + str(mrr))
        peer_class_2_tail_mrr[peers_class] = mrr

    else:
        print("\t" + peers_class + ": UNDEFINED")
        peer_class_2_tail_mrr[peers_class] = "--"
print()




print("\n\n\n")

for prediction_type in ["head", "tail"]:
    print("MRR for " + prediction_type + " predictions:")
    for head_peers_interval in PEER_INTERVALS:
        row_mrr = []

        for tail_peers_interval in PEER_INTERVALS:
            head_peers_interval_lower_bound = str(head_peers_interval[0])
            head_peers_interval_upper_bound = str(head_peers_interval[1])
            tail_peers_interval_lower_bound = str(tail_peers_interval[0])
            tail_peers_interval_upper_bound = str(tail_peers_interval[1])
            peers_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

            if prediction_type == "head":
                mrr = peer_class_2_head_mrr[peers_class]
            else:
                mrr = peer_class_2_tail_mrr[peers_class]

            row_mrr.append(str(mrr))
        print("; ".join(row_mrr).replace('.', ','))
    print()
