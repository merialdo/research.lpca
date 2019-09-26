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


dataset_name = datasets.FB15K
rel_2_mentions = relation_mentions.read(dataset_name)

model_entries = read_filtered_ranks_entries_for(models.ROTATE, dataset_name)
test_fact_2_class = peer_classes.read(dataset_name, return_fact_2_class=True)

top_100_rels = sorted(rel_2_mentions.items(), key=lambda x:x[1], reverse=True)[0:100]
print(top_100_rels)


for rel, mentions in top_100_rels:

    print("Relation name: " +  rel)
    print("Relation mentions: " + str(mentions))

    overall_all = 0.0
    peers_class_2_overall_counts = dict()
    for fine_class in PEER_CLASSES:
        peers_class_2_overall_counts[fine_class] = 0.0

    for entry in model_entries:
        head, relation, tail = entry['head'], entry['relation'], entry['tail']

        if relation != rel:
            continue

        fine_class = test_fact_2_class[";".join([head, relation, tail])]
        overall_all += 1
        peers_class_2_overall_counts[fine_class] += 1

    print("Relation percentages in test set")
    print("              (0, 1)\t(1, 4)\t\t(4, 25)\t(25, 100)\t(100, 'inf')")
    for head_peers_interval in PEER_INTERVALS:
        row_percentages = []

        for tail_peers_interval in PEER_INTERVALS:
            head_peers_interval_lower_bound = str(head_peers_interval[0])
            head_peers_interval_upper_bound = str(head_peers_interval[1])
            tail_peers_interval_lower_bound = str(tail_peers_interval[0])
            tail_peers_interval_upper_bound = str(tail_peers_interval[1])
            peers_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

            if overall_all == 0:
                row_percentages.append("0,00%")
            else:
                overall_percentage = float(peers_class_2_overall_counts[peers_class]) * 100 / float(overall_all)
                row_percentages.append(str(round(overall_percentage, 1)) + "%")
        prefix = str(head_peers_interval)
        prefix_len = len(prefix)
        if 14 - prefix_len > 0:
            spaces = " " * (14 - prefix_len)
            prefix += spaces

        print(prefix + "\t\t".join(row_percentages).replace('.', ','))
    print()


    peers_class_2_head_ranks = defaultdict(lambda:[])
    peers_class_2_tail_ranks = defaultdict(lambda:[])

    for entry in model_entries:

        head = entry['head']
        relation = entry['relation']
        tail = entry['tail']
        head_rank_filtered = entry['head_rank_filtered']
        tail_rank_filtered = entry['tail_rank_filtered']

        if relation != rel:
            continue

        peers_class = test_fact_2_class[";".join([head, relation, tail])]

        peers_class_2_head_ranks[peers_class].append(float(head_rank_filtered))
        peers_class_2_tail_ranks[peers_class].append(float(tail_rank_filtered))


    peer_class_2_head_mrr = defaultdict(lambda: 0)
    peer_class_2_tail_mrr = defaultdict(lambda: 0)

    for peers_class in PEER_CLASSES:
        if peers_class_2_overall_counts[peers_class] > 0:
            head_ranks = peers_class_2_head_ranks[peers_class]
            inverse_ranks = [1.0/x for x in head_ranks]
            mrr = numpy.average(inverse_ranks)
            mrr = round(mrr, 2)
            peer_class_2_head_mrr[peers_class] = mrr
        else:
            peer_class_2_head_mrr[peers_class] = "----"

    for peers_class in PEER_CLASSES:
        if peers_class_2_overall_counts[peers_class] > 0:
            tail_ranks = peers_class_2_tail_ranks[peers_class]
            inverse_ranks = [1.0 / x for x in tail_ranks]
            mrr = numpy.average(inverse_ranks)
            mrr = round(mrr, 2)
            peer_class_2_tail_mrr[peers_class] = mrr

        else:
            peer_class_2_tail_mrr[peers_class] = "----"

    print()




    for prediction_type in ["head", "tail"]:

        print("MRR for " + prediction_type + " predictions:")

        print("              (0, 1)\t(1, 4)\t\t(4, 25)\t(25, 100)\t(100, 'inf')")
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

                mrr = str(mrr)
                if len(mrr) < 4:
                    mrr += "0" *  (4-len(mrr))

                row_mrr.append(mrr)

            prefix = str(head_peers_interval)
            prefix_len = len(prefix)
            if 14 - prefix_len > 0:
                spaces = " " * (14 - prefix_len)
                prefix += spaces

            print(prefix + "\t\t".join(row_mrr).replace('.', ','))
        print()



    print("\n\n\n")
