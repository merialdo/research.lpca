import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.relation_cardinalities import relation_coarse_classes
from datasets import FB15K
from io_utils import *
from models import ROTATE

model_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)

entity_2_in_degree, entity_2_out_degree, entity_2_degree = entity_degrees.read(FB15K)
relation_2_coarse_class = relation_coarse_classes.read(FB15K, return_rel_2_class=True)

one_to_one_entries = []
one_to_many_entries = []
many_to_one_entries = []
many_to_many_entries = []

overall_all = 0.0
for entry in model_entries:
    relation = entry['relation']

    overall_all += 1
    if relation_2_coarse_class[relation] == "one to one":
        one_to_one_entries.append(entry)
    elif relation_2_coarse_class[relation] == "one to many":
        one_to_many_entries.append(entry)
    elif relation_2_coarse_class[relation] == "many to one":
        many_to_one_entries.append(entry)
    else:
        many_to_many_entries.append(entry)

# plot in_degree 2 ranks of head predictions and in_degree 2 rank of tail predictions
def plot_ranks_by_in_degree(entries, entity_2_in_degree, reltype):
    in_degree_heads, in_degree_tails = [], []
    head_predictions_ranks, tail_predictions_ranks = [], []

    for entry in entries:
        in_degree_heads.append(entity_2_in_degree[entry["head"]])
        head_predictions_ranks.append(entry["head_rank_filtered"])

        in_degree_tails.append(entity_2_in_degree[entry["tail"]])
        tail_predictions_ranks.append(entry["tail_rank_filtered"])

    print(reltype + ": head predictions mean rank: " + str(np.average(head_predictions_ranks)))
    print(reltype + ": tail predictions mean rank: " + str(np.average(tail_predictions_ranks)))

    plt.scatter(in_degree_heads, head_predictions_ranks, s=1, color='blue')
    plt.title(reltype + ": in-degree vs rank in head predictions")
    plt.xlabel("in degree of the head entity of each head prediction")
    plt.ylabel("filtered rank for each head prediction in which the head entity to predict has that in degree")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


    plt.scatter(in_degree_tails, tail_predictions_ranks, s=1, color='blue')
    plt.title(reltype + ": in-degree vs rank in tail predictions")
    plt.xlabel("in degree of the tail entity of each tail prediction")
    plt.ylabel("filtered rank for each tail prediction in which the tail entity to predict has that in degree")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


# plot in_degree 2 ranks of head predictions and in_degree 2 rank of tail predictions
def plot_ranks_by_out_degree(entries, entity_2_out_degree, reltype):
    out_degree_heads, out_degree_tails = [], []
    head_predictions_ranks, tail_predictions_ranks = [], []

    for entry in entries:
        out_degree_heads.append(entity_2_out_degree[entry["head"]])
        head_predictions_ranks.append(entry["head_rank_filtered"])

        out_degree_tails.append(entity_2_out_degree[entry["tail"]])
        tail_predictions_ranks.append(entry["tail_rank_filtered"])

    plt.scatter(out_degree_heads, head_predictions_ranks, s=1, color='blue')
    plt.title(reltype + ": out-degree vs rank in head predictions")
    plt.xlabel("out degree of the head entity of each head prediction")
    plt.ylabel("filtered rank for each head prediction in which the head entity to predict has that out-degree")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


    plt.scatter(out_degree_tails, tail_predictions_ranks, s=1, color='blue')
    plt.title(reltype + ": out-degree vs rank in tail predictions")
    plt.xlabel("out-degree of the tail entity of each tail prediction")
    plt.ylabel("filtered rank for each tail prediction in which the tail entity to predict has that out-degree")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()



plot_ranks_by_in_degree(one_to_one_entries, entity_2_in_degree, "One to one")
plot_ranks_by_in_degree(one_to_many_entries, entity_2_in_degree, "One to many")
plot_ranks_by_in_degree(many_to_one_entries, entity_2_in_degree, "Many to one")
plot_ranks_by_in_degree(many_to_many_entries, entity_2_in_degree, "Many to many")

plot_ranks_by_out_degree(one_to_one_entries, entity_2_out_degree, "One to one")
plot_ranks_by_out_degree(one_to_many_entries, entity_2_out_degree, "One to many")
plot_ranks_by_out_degree(many_to_one_entries, entity_2_out_degree, "Many to one")
plot_ranks_by_out_degree(many_to_many_entries, entity_2_out_degree, "Many to many")