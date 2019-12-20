import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.peers import peers
from datasets import FB15K

from io_utils import *
from models import TRANSE, ROTATE, CONVE, SIMPLE, ANYBURL


def plot_dict(dict, title, xlabel, ylabel):
    x = []
    y = []

    for item in (sorted(dict.items(), key=lambda x: x[0])):
        x.append(item[0])
        y.append(item[1])

    plt.scatter(x, y, s=1, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


def analyze(model_entries, head_prediction_2_peers, tail_prediction_2_peers):

    head_peers_number_2_head_hits1 = defaultdict(lambda: 0.0)
    tail_peers_number_2_head_hits1 = defaultdict(lambda: 0.0)
    head_peers_number_2_tail_hits1 = defaultdict(lambda: 0.0)
    tail_peers_number_2_tail_hits1 = defaultdict(lambda: 0.0)

    # these dicts map
    #   - each number of peers (head peers or tail peers respectively)
    #   - to the number of predictions in the test set that have that number of peers (head peers or tail peers respectively)
    # This is then used to compute the percentage of correct predictions with a certain number of peers in the head (or in the tail respectively)
    head_peers_number_2_count = defaultdict(lambda: 0.0)
    tail_peers_number_2_count = defaultdict(lambda: 0.0)


    for entry in model_entries:
        head = entry["head"]
        relation = entry["relation"]
        tail = entry["tail"]

        head_rank = int(entry["head_rank_filtered"])
        tail_rank = int(entry["tail_rank_filtered"])

        head_peers = head_prediction_2_peers[relation + ";" + tail]
        tail_peers = tail_prediction_2_peers[head + ";" + relation]

        if head_rank == 1:
            head_peers_number_2_head_hits1[head_peers] += 1
            tail_peers_number_2_head_hits1[tail_peers] += 1

        if tail_rank == 1:
            head_peers_number_2_tail_hits1[head_peers] += 1
            tail_peers_number_2_tail_hits1[tail_peers] += 1

        head_peers_number_2_count[head_peers] += 1
        tail_peers_number_2_count[tail_peers] += 1


    # switch to percentages
    for key in head_peers_number_2_head_hits1:
        head_peers_number_2_head_hits1[key] = float(head_peers_number_2_head_hits1[key]) / float(head_peers_number_2_count[key])
        head_peers_number_2_tail_hits1[key] = float(head_peers_number_2_tail_hits1[key]) / float(head_peers_number_2_count[key])

    for key in tail_peers_number_2_head_hits1:
        tail_peers_number_2_head_hits1[key] = float(tail_peers_number_2_head_hits1[key]) / float(tail_peers_number_2_count[key])
        tail_peers_number_2_tail_hits1[key] = float(tail_peers_number_2_tail_hits1[key]) / float(tail_peers_number_2_count[key])


    return head_peers_number_2_head_hits1, tail_peers_number_2_head_hits1, head_peers_number_2_tail_hits1, tail_peers_number_2_tail_hits1


head_prediction_2_peers, tail_prediction_2_peers = peers.read(FB15K)

transE_entries = performances.read_filtered_ranks_entries_for(TRANSE, FB15K)
rotatE_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)
convE_entries = performances.read_filtered_ranks_entries_for(CONVE, FB15K)
simplE_entries = performances.read_filtered_ranks_entries_for(SIMPLE, FB15K)
anyburl_entries = performances.read_filtered_ranks_entries_for(ANYBURL, FB15K)


rotatE_h_peers_2_h_hits1, rotatE_t_peers_2_h_hits1, rotatE_h_peers_2_t_hits1, rotatE_t_peers_2_t_hits1 = analyze(rotatE_entries, head_prediction_2_peers, tail_prediction_2_peers)


plot_dict(rotatE_h_peers_2_h_hits1, "RotatE head peers vs head hits@1", "head peers", "percentage of hits@1 on all head predictions of entities with that number of head peers")
plot_dict(rotatE_t_peers_2_h_hits1, "RotatE tail peers vs head hits@1", "tail peers", "percentage of hits@1 on all head predictions of entities with that number of tail peers")
plot_dict(rotatE_h_peers_2_t_hits1, "RotatE head peers vs tail hits@1", "head peers", "percentage of hits@1 on all tail predictions of entities with that number of head peers")
plot_dict(rotatE_t_peers_2_t_hits1, "RotatE tail peers vs tail hits@1", "tail peers", "percentage of hits@1 on all tail predictions of entities with that number of tail peers")

transE_h_peers_2_h_hits1, transE_t_peers_2_h_hits1, transE_h_peers_2_t_hits1, transE_t_peers_2_t_hits1 = analyze(transE_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_dict(transE_h_peers_2_h_hits1, "TransE head peers vs head hits@1", "head peers", "percentage of hits@1 on all head predictions of entities with that number of head peers")
plot_dict(transE_t_peers_2_h_hits1, "TransE tail peers vs head hits@1", "tail peers", "percentage of hits@1 on all head predictions of entities with that number of tail peers")
plot_dict(transE_h_peers_2_t_hits1, "TransE head peers vs tail hits@1", "head peers", "percentage of hits@1 on all tail predictions of entities with that number of head peers")
plot_dict(transE_t_peers_2_t_hits1, "TransE tail peers vs tail hits@1", "tail peers", "percentage of hits@1 on all tail predictions of entities with that number of tail peers")
