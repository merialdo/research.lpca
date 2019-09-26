import numpy
import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.peers import peers
from datasets import FB15K

from io_utils import *
from models import TRANSE, ROTATE, CONVE, SIMPLE, ANYBURL, TUCKER


def plot_couples(couples, title, xlabel, ylabel):
    x = []
    y = []

    for couple in (sorted(couples, key=lambda x: x[0])):
        x.append(couple[0])
        y.append(couple[1])

    plt.scatter(x, y, s=1, color='blue', alpha=0.4)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


def analyze(model_entries, head_prediction_2_peers, tail_prediction_2_peers):

    head_peers_head_ranks = []
    tail_peers_head_ranks = []
    head_peers_tail_ranks = []
    tail_peers_tail_ranks = []

    for entry in model_entries:
        head = entry["head"]
        relation = entry["relation"]
        tail = entry["tail"]

        head_rank = int(entry["head_rank_filtered"])
        tail_rank = int(entry["tail_rank_filtered"])

        head_peers = head_prediction_2_peers[relation + ";" + tail]
        tail_peers = tail_prediction_2_peers[head + ";" + relation]

        head_peers_head_ranks.append((head_peers, head_rank))
        tail_peers_head_ranks.append((tail_peers, head_rank))
        head_peers_tail_ranks.append((head_peers, tail_rank))
        tail_peers_tail_ranks.append((tail_peers, tail_rank))

    return head_peers_head_ranks, tail_peers_head_ranks, head_peers_tail_ranks, tail_peers_tail_ranks


head_prediction_2_peers, tail_prediction_2_peers = peers.read(FB15K)

transE_entries = performances.read_filtered_ranks_entries_for(TRANSE, FB15K)
rotatE_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)
tuckER_entries = performances.read_filtered_ranks_entries_for(TUCKER, FB15K)
convE_entries = performances.read_filtered_ranks_entries_for(CONVE, FB15K)
simplE_entries = performances.read_filtered_ranks_entries_for(SIMPLE, FB15K)
anyburl_entries = performances.read_filtered_ranks_entries_for(ANYBURL, FB15K)


rotatE_h_peers_h_rank, rotatE_t_peers_h_rank, rotatE_h_peers_t_rank, rotatE_t_peers_t_rank = analyze(rotatE_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_couples(rotatE_h_peers_h_rank, "RotatE head peers vs head rank", "head peers", "rank of all head predictions of entities with that number of head peers")
plot_couples(rotatE_t_peers_h_rank, "RotatE tail peers vs head rank", "tail peers", "rank of all head predictions of entities with that number of tail peers")
plot_couples(rotatE_h_peers_t_rank, "RotatE head peers vs tail rank", "head peers", "rank of all tail predictions of entities with that number of head peers")
plot_couples(rotatE_t_peers_t_rank, "RotatE tail peers vs tail rank", "tail peers", "rank of all tail predictions of entities with that number of tail peers")


tuckER_h_peers_h_rank, tuckER_t_peers_h_rank, tuckER_h_peers_t_rank, tuckER_t_peers_t_rank = analyze(tuckER_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_couples(tuckER_h_peers_h_rank, "TuckER head peers vs head rank", "head peers", "rank of all head predictions of entities with that number of head peers")
plot_couples(tuckER_t_peers_h_rank, "TuckER tail peers vs head rank", "tail peers", "rank of all head predictions of entities with that number of tail peers")
plot_couples(tuckER_h_peers_t_rank, "TuckER head peers vs tail rank", "head peers", "rank of all tail predictions of entities with that number of head peers")
plot_couples(tuckER_t_peers_t_rank, "TuckER tail peers vs tail rank", "tail peers", "rank of all tail predictions of entities with that number of tail peers")



transE_h_peers_h_rank, transE_t_peers_h_rank, transE_h_peers_t_rank, transE_t_peers_t_rank = analyze(transE_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_couples(transE_h_peers_h_rank, "TransE head peers vs head rank", "head peers", "rank of all head predictions of entities with that number of head peers")
plot_couples(transE_t_peers_h_rank, "TransE tail peers vs head rank", "tail peers", "rank of all head predictions of entities with that number of tail peers")
plot_couples(transE_h_peers_t_rank, "TransE head peers vs tail rank", "head peers", "rank of all tail predictions of entities with that number of head peers")
plot_couples(transE_t_peers_t_rank, "TransE tail peers vs tail rank", "tail peers", "rank of all tail predictions of entities with that number of tail peers")
