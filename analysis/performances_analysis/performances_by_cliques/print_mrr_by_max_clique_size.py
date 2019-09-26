import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

import datasets
import models
from dataset_analysis.cliques import cliques
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.siblings.sibling_classes import SIBLING_CLASSES, INTERVALS
from dataset_analysis.siblings import sibling_classes
from io_utils import *
from performances import read_filtered_ranks_entries_for


dataset_name = datasets.FB15K
model_entries = read_filtered_ranks_entries_for(models.TUCKER, dataset_name)

test_fact_2_clique_size = cliques.read(dataset_name, return_fact_2_clique_size=True)
clique_size_2_test_facts = cliques.read(dataset_name)

max_clique_sizes = sorted(clique_size_2_test_facts.keys())

# === count and print the percentage of facts in each sibling class ===
overall_all = 0.0
clique_size_2_number_of_facts = dict()
for clique_size in clique_size_2_test_facts:
    number_of_facts = len(clique_size_2_test_facts[clique_size])
    overall_all += number_of_facts
    clique_size_2_number_of_facts[clique_size] = number_of_facts

clique_size_2_percentage_of_facts = dict()
for clique_size in clique_size_2_test_facts:
    clique_size_2_percentage_of_facts[clique_size] = float(clique_size_2_number_of_facts[clique_size]) / float(overall_all)


# print the percentage of facts in each sibling class
print("Percentages of test facts grouped by the size of the max clique that they belong to")

result = []
for clique_size in max_clique_sizes:
    percentage = clique_size_2_percentage_of_facts[clique_size]
    result.append(str(round(100*percentage, 2)))
print(";".join(result))

max_clique_size_2_head_ranks = defaultdict(lambda:[])
max_clique_size_2_tail_ranks = defaultdict(lambda:[])

for entry in model_entries:

    head = entry['head']
    relation = entry['relation']
    tail = entry['tail']
    head_rank_filtered = entry['head_rank_filtered']
    tail_rank_filtered = entry['tail_rank_filtered']
    max_clique_size = test_fact_2_clique_size[";".join([head, relation, tail])]

    max_clique_size_2_head_ranks[max_clique_size].append(float(head_rank_filtered))
    max_clique_size_2_tail_ranks[max_clique_size].append(float(tail_rank_filtered))

max_clique_size_2_head_mrr = defaultdict(lambda: 0)
max_clique_size_2_tail_mrr = defaultdict(lambda: 0)


print("MRR in head predictions: ")
for max_clique_size in max_clique_sizes:
    if clique_size_2_number_of_facts[max_clique_size] > 0:
        head_ranks = max_clique_size_2_head_ranks[max_clique_size]
        inverse_ranks = [1.0/x for x in head_ranks]
        mrr = numpy.average(inverse_ranks)
        mrr = round(mrr, 2)
        print("\t" + str(max_clique_size) + ": " + str(mrr))
        max_clique_size_2_head_mrr[max_clique_size] = mrr
    else:
        print("\t" + str(max_clique_size) + ": UNDEFINED")
        max_clique_size_2_head_mrr[max_clique_size] = "--"
print()

for max_clique_size in max_clique_sizes:
    if clique_size_2_number_of_facts[max_clique_size] > 0:
        tail_ranks = max_clique_size_2_tail_ranks[max_clique_size]
        inverse_ranks = [1.0 / x for x in tail_ranks]
        mrr = numpy.average(inverse_ranks)
        mrr = round(mrr, 2)
        print("\t" + str(max_clique_size) + ": " + str(mrr))
        max_clique_size_2_tail_mrr[max_clique_size] = mrr
    else:
        print("\t" + str(max_clique_size) + ": UNDEFINED")
        max_clique_size_2_tail_mrr[max_clique_size] = "--"
print()




print("\n\n\n")

for prediction_type in ["head", "tail"]:
    print("MRR for " + prediction_type + " predictions:")

    row_mrr = []
    for max_clique_size in max_clique_sizes:
        if prediction_type == "head":
            mrr = max_clique_size_2_head_mrr[max_clique_size]
        else:
            mrr = max_clique_size_2_tail_mrr[max_clique_size]
        row_mrr.append(str(mrr))
    print("; ".join(row_mrr).replace('.', ','))
print()
