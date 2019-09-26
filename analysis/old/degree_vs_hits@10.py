import numpy as np
import matplotlib.pyplot as plt
from io_utils import *
from collections import defaultdict


def plot_dict(dict, title, xlabel, ylabel, xticks=[]):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, s=1)
    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.xscale('log')


    plt.grid(True)


    plt.show()


def plot_dicts(dict1, dict2, title, xlabel, ylabel, xticks=[]):
    x1 = []
    y1 = []
    for item in (sorted(dict1.items(), key=lambda x: x[0])):
        x1.append(item[0])
        y1.append(item[1])

    plt.scatter(x1, y1, s=1)

    x2 = []
    y2 = []
    for item in (sorted(dict2.items(), key=lambda x: x[0])):
        x2.append(item[0])
        y2.append(item[1])

    plt.scatter(x2, y2, s=1)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.xscale('log')


    plt.grid(True)


    plt.show()



def get_h10_dicts_from_entries(entries):

    head_degree_2_head_h10 = defaultdict(lambda:0)
    head_degree_2_tail_h10 = defaultdict(lambda:0)

    tail_degree_2_head_h10 = defaultdict(lambda:0)
    tail_degree_2_tail_h10 = defaultdict(lambda:0)

    relation_mentions_2_head_h10 = defaultdict(lambda:0)
    relation_mentions_2_tail_h10 = defaultdict(lambda:0)

    head_degree_2_count = defaultdict(lambda:0)
    tail_degree_2_count = defaultdict(lambda:0)
    relation_mentions_2_count = defaultdict(lambda:0)

    all_head_degrees = set()
    all_tail_degrees = set()
    all_relation_mentions = set()

    for entry in entries:

        head_degree = int(entry["head_degree"])
        tail_degree = int(entry["tail_degree"])
        relation_mentions = int(entry["relation_mentions"])
        head_rank = int(entry["head_rank_raw"])
        tail_rank = int(entry["tail_rank_raw"])

        all_head_degrees.add(head_degree)
        all_tail_degrees.add(tail_degree)
        all_relation_mentions.add(relation_mentions)

        head_degree_2_count[head_degree] += 1
        tail_degree_2_count[tail_degree] += 1
        relation_mentions_2_count[relation_mentions] += 1

        if head_rank <= 10:
            head_degree_2_head_h10[head_degree] += 1
            tail_degree_2_head_h10[tail_degree] += 1
            relation_mentions_2_head_h10[relation_mentions] += 1
        if tail_rank <= 10:
            head_degree_2_tail_h10[head_degree] += 1
            tail_degree_2_tail_h10[tail_degree] += 1
            relation_mentions_2_tail_h10[relation_mentions] += 1

    head_degree_2_head_h10_perc = dict()
    head_degree_2_tail_h10_perc = dict()
    tail_degree_2_head_h10_perc = dict()
    tail_degree_2_tail_h10_perc = dict()
    relation_mentions_2_head_h10_perc = dict()
    relation_mentions_2_tail_h10_perc = dict()

    for head_degree in all_head_degrees:
        head_degree_2_head_h10_perc[head_degree] = float(head_degree_2_head_h10[head_degree])/float(head_degree_2_count[head_degree])
        head_degree_2_tail_h10_perc[head_degree] = float(head_degree_2_tail_h10[head_degree])/float(head_degree_2_count[head_degree])
    for tail_degree in all_tail_degrees:
        tail_degree_2_head_h10_perc[tail_degree] = float(tail_degree_2_head_h10[tail_degree])/float(tail_degree_2_count[tail_degree])
        tail_degree_2_tail_h10_perc[tail_degree] = float(tail_degree_2_tail_h10[tail_degree])/float(tail_degree_2_count[tail_degree])
    for relation_mentions in all_relation_mentions:
        relation_mentions_2_head_h10_perc[relation_mentions] = float(relation_mentions_2_head_h10[relation_mentions])/float(relation_mentions_2_count[relation_mentions])
        relation_mentions_2_tail_h10_perc[relation_mentions] = float(relation_mentions_2_tail_h10[relation_mentions])/float(relation_mentions_2_count[relation_mentions])



    return head_degree_2_head_h10_perc, head_degree_2_tail_h10_perc, \
           tail_degree_2_head_h10_perc, tail_degree_2_tail_h10_perc, \
           relation_mentions_2_head_h10_perc, relation_mentions_2_tail_h10_perc



transe_entries = get_entries_from_file("/Users/andrea/paper/FB15K/transe_fb15k_test_with_correct_ranks.csv")
distmult_entries = get_entries_from_file("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")

transe_hxh, transe_hxt, transe_txh, transe_txt, transe_rxh, transe_rxt = get_h10_dicts_from_entries(transe_entries)
distmult_hxh, distmult_hxt, distmult_txh, distmult_txt, distmult_rxh, distmult_rxt = get_h10_dicts_from_entries(distmult_entries)



plot_dict(transe_hxh, "TransE head degree vs head hits@10", "head degree", "hits@10 for the head of test facts with that head degree")
plot_dict(transe_hxt, "TransE head degree vs tail hits@10", "head degree", "hits@10 for the tail of test facts with that head degree")
plot_dict(transe_txh, "TransE tail degree vs head hits@10", "tail degree", "hits@10 for the head of test facts with that tail degree")
plot_dict(transe_txt, "TransE tail degree vs tail hits@10", "tail degree", "hits@10 for the tail for test facts with that tail degree")
plot_dict(transe_rxh, "TransE relation mentions vs head hits@10", "relation mentions", "hits@10 for the head of test facts with that relation mentions")
plot_dict(transe_rxt, "TransE relation mentions vs tail hits@10", "relation mentions", "hits@10 for the tail for test facts with that relation mentions")

plot_dict(distmult_hxh, "DistMult head degree vs head hits@10", "head degree", "hits@10 of head for test facts with that head degree")
plot_dict(distmult_hxt, "DistMult head degree vs tail hits@10", "head degree", "hits@10 of tail for test facts with that head degree")
plot_dict(distmult_txh, "DistMult tail degree vs head hits@10", "tail degree", "hits@10 of head of test facts with that tail degree")
plot_dict(distmult_txt, "DistMult tail degree vs tail hits@10", "tail degree", "hits@10 of tail for test facts with that tail degree")
plot_dict(distmult_rxh, "DistMult relation mentions vs head hits@10", "relation mentions", "hits@10 for the head of test facts with that relation mentions")
plot_dict(distmult_rxt, "DistMult relation mentions vs tail hits@10", "relation mentions", "hits@10 for the tail for test facts with that relation mentions")

plot_dicts(transe_hxh, distmult_hxh, "DistMult relation mentions vs tail hits@10", "relation mentions", "hits@10 for the tail for test facts with that relation mentions")


