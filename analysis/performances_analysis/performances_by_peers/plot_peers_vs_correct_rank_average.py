import numpy
import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.peers import peers
from datasets import FB15K, Dataset

from io_utils import *
from models import TRANSE, ROTATE, TUCKER, CONVE, SIMPLE, ANYBURL


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

    head_peers_number_2_head_ranks = defaultdict(lambda: [])
    tail_peers_number_2_head_ranks = defaultdict(lambda: [])
    head_peers_number_2_tail_ranks = defaultdict(lambda: [])
    tail_peers_number_2_tail_ranks = defaultdict(lambda: [])

    for entry in model_entries:
        head = entry["head"]
        relation = entry["relation"]
        tail = entry["tail"]

        head_rank = int(entry["head_rank_filtered"])
        tail_rank = int(entry["tail_rank_filtered"])

        head_peers = head_prediction_2_peers[relation + ";" + tail]
        tail_peers = tail_prediction_2_peers[head + ";" + relation]

        head_peers_number_2_head_ranks[head_peers].append(head_rank)
        tail_peers_number_2_head_ranks[tail_peers].append(head_rank)
        head_peers_number_2_tail_ranks[head_peers].append(tail_rank)
        tail_peers_number_2_tail_ranks[tail_peers].append(tail_rank)

    for key in head_peers_number_2_head_ranks:
        head_peers_number_2_head_ranks[key] = numpy.average(head_peers_number_2_head_ranks[key])
        head_peers_number_2_tail_ranks[key] = numpy.average(head_peers_number_2_tail_ranks[key])
    for key in tail_peers_number_2_head_ranks:
        tail_peers_number_2_head_ranks[key] = numpy.average(tail_peers_number_2_head_ranks[key])
        tail_peers_number_2_tail_ranks[key] = numpy.average(tail_peers_number_2_tail_ranks[key])

    return head_peers_number_2_head_ranks, tail_peers_number_2_head_ranks, head_peers_number_2_tail_ranks, tail_peers_number_2_tail_ranks


head_prediction_2_peers, tail_prediction_2_peers = peers.read(FB15K)

test_triples = Dataset(FB15K).test_triples

transE_entries = performances.read_filtered_ranks_entries_for(TRANSE, FB15K)
rotatE_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)
tuckER_entries = performances.read_filtered_ranks_entries_for(TUCKER, FB15K)
convE_entries = performances.read_filtered_ranks_entries_for(CONVE, FB15K)
simplE_entries = performances.read_filtered_ranks_entries_for(SIMPLE, FB15K)
anyburl_entries = performances.read_filtered_ranks_entries_for(ANYBURL, FB15K)


rotatE_h_peers_2_h_rank, rotatE_t_peers_2_h_rank, rotatE_h_peers_2_t_rank, rotatE_t_peers_2_t_rank = analyze(rotatE_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_dict(rotatE_h_peers_2_h_rank, "RotatE head peers vs head mean rank", "head peers", "mean rank of all head predictions of entities with that number of head peers")
plot_dict(rotatE_t_peers_2_h_rank, "RotatE tail peers vs head mean rank", "tail peers", "mean rank of all head predictions of entities with that number of tail peers")
plot_dict(rotatE_h_peers_2_t_rank, "RotatE head peers vs tail mean rank", "head peers", "mean rank of all tail predictions of entities with that number of head peers")
plot_dict(rotatE_t_peers_2_t_rank, "RotatE tail peers vs tail mean rank", "tail peers", "mean rank of all tail predictions of entities with that number of tail peers")


tuckER_h_peers_2_h_rank, tuckER_t_peers_2_h_rank, tuckER_h_peers_2_t_rank, tuckER_t_peers_2_t_rank = analyze(tuckER_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_dict(tuckER_h_peers_2_h_rank, "TuckER head peers vs head mean rank", "head peers", "mean rank of all head predictions of entities with that number of head peers")
plot_dict(tuckER_t_peers_2_h_rank, "TuckER tail peers vs head mean rank", "tail peers", "mean rank of all head predictions of entities with that number of tail peers")
plot_dict(tuckER_h_peers_2_t_rank, "TuckER head peers vs tail mean rank", "head peers", "mean rank of all tail predictions of entities with that number of head peers")
plot_dict(tuckER_t_peers_2_t_rank, "TuckER tail peers vs tail mean rank", "tail peers", "mean rank of all tail predictions of entities with that number of tail peers")



transE_h_peers_2_h_rank, transE_t_peers_2_h_rank, transE_h_peers_2_t_rank, transE_t_peers_2_t_rank = analyze(transE_entries, head_prediction_2_peers, tail_prediction_2_peers)

plot_dict(transE_h_peers_2_h_rank, "TransE head peers vs head mean rank", "head peers", "mean rank of all head predictions of entities with that number of head peers")
plot_dict(transE_t_peers_2_h_rank, "TransE tail peers vs head mean rank", "tail peers", "mean rank of all head predictions of entities with that number of tail peers")
plot_dict(transE_h_peers_2_t_rank, "TransE head peers vs tail mean rank", "head peers", "mean rank of all tail predictions of entities with that number of head peers")
plot_dict(transE_t_peers_2_t_rank, "TransE tail peers vs tail mean rank", "tail peers", "mean rank of all tail predictions of entities with that number of tail peers")
