import html
import os

import numpy
# read entities and create a map mid -> id
import datasets
import models
from dataset_analysis.degrees import entity_degrees


def print_metrics_for(model_name, dataset_name):
    filtered_ranks_path = models.filtered_ranks_path(model_name, dataset_name)

    if not os.path.isfile(filtered_ranks_path):
        return

    with open(filtered_ranks_path, "r") as filtered_ranks_file:

        lines = filtered_ranks_file.readlines()

        all_ranks = []
        for line in lines:
            line = html.unescape(line.strip())  # the unescape is necessary because some elements in YAGO3 have '&amp;'

            head, relation, tail, head_rank, tail_rank = line.strip().split(";")

            # if "." in relation:
            #   continue

            if head_rank.startswith("MISS"):
                head_rank = len(entity_2_degree)
            all_ranks.append(float(head_rank))

            if tail_rank.startswith("MISS"):
                tail_rank = (len(entity_2_degree))
            all_ranks.append(float(tail_rank))

        hits_1 = 0
        hits_3 = 0
        hits_5 = 0
        hits_10 = 0

        for rank in all_ranks:
            if rank == 1:
                hits_1 += 1
            if rank <= 3:
                hits_3 += 1
            if rank <= 5:
                hits_5 += 1
            if rank <= 10:
                hits_10 += 1

        print("\nResults for " + model_name + " on " + dataset_name + ":")

        mean_rank = numpy.average([float(x) for x in all_ranks])
        mean_reciprocal_rank = numpy.average([1.0 / float(x) for x in all_ranks])
        hits_1_perc = float(hits_1) * 100 / len(all_ranks)
        hits_3_perc = float(hits_3) * 100 / len(all_ranks)
        hits_5_perc = float(hits_5) * 100 / len(all_ranks)
        hits_10_perc = float(hits_10) * 100 / len(all_ranks)

        print("Mean Rank:\t\t\t\t\t%f" % mean_rank)
        print("Mean Reciprocal Rank: \t\t%f" % mean_reciprocal_rank)
        print("Hits@1:\t\t\t\t\t\t%f%%" % hits_1_perc)
        print("Hits@3:\t\t\t\t\t\t%f%%" % hits_3_perc)
        print("Hits@5:\t\t\t\t\t\t%f%%" % hits_5_perc)
        print("Hits@10:\t\t\t\t\t%f%%" % hits_10_perc)
entity_2_in_degree, entity_2_out_degree, entity_2_degree = entity_degrees.read(datasets.FB15K)

# print_metrics_for(models.RSN, datasets.FB15K)

for cur_dataset_name in datasets.ALL_DATASET_NAMES:
    for cur_model_name in [models.COMPLEX]: #models.ALL_NAMES:
        print_metrics_for(cur_model_name, cur_dataset_name)
        print()


