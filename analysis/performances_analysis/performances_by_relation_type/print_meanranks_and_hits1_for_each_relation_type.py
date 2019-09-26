import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees, relation_mentions
from dataset_analysis.relation_cardinalities import relation_coarse_classes
from datasets import FB15K
from io_utils import *
from models import TRANSE, ROTATE, CONVE, SIMPLE, ANYBURL

transE_entries = performances.read_filtered_ranks_entries_for(TRANSE, FB15K)
rotatE_entries = performances.read_filtered_ranks_entries_for(ROTATE, FB15K)
convE_entries = performances.read_filtered_ranks_entries_for(CONVE, FB15K)
simplE_entries = performances.read_filtered_ranks_entries_for(SIMPLE, FB15K)
anyburl_entries = performances.read_filtered_ranks_entries_for(ANYBURL, FB15K)

entity_2_in_degree, entity_2_out_degree, entity_2_degree = entity_degrees.read(FB15K)
relation2mentions = relation_mentions.read(FB15K)
relation_2_type = relation_coarse_classes.read(FB15K, return_rel_2_class=True)



model_name_2_entries = {"TransE" : transE_entries,
                        "ConvE" : convE_entries,
                        "SimplE" : simplE_entries,
                        "RotatE" : rotatE_entries,
                        "AnyBURL" : anyburl_entries}

print("RANKS ANALYSIS")
for model_name in model_name_2_entries:

    print(model_name)

    one_to_one_head_ranks, all_one_to_one_head_predictions = [], 0
    one_to_one_tail_ranks, all_one_to_one_tail_predictions = [], 0
    one_to_many_head_ranks, all_one_to_many_head_predictions = [], 0
    one_to_many_tail_ranks, all_one_to_many_tail_predictions = [], 0
    many_to_one_head_ranks, all_many_to_one_head_predictions = [], 0
    many_to_one_tail_ranks, all_many_to_one_tail_predictions = [], 0
    many_to_many_head_ranks, all_many_to_many_head_predictions = [], 0
    many_to_many_tail_ranks, all_many_to_many_tail_predictions = [], 0

    for entry in model_name_2_entries[model_name]:
        relation = entry['relation']

        if relation_2_type[relation] == "one to one":
            all_one_to_one_head_predictions += 1
            all_one_to_one_tail_predictions += 1
            one_to_one_head_ranks.append(entry["head_rank_filtered"])
            one_to_one_tail_ranks.append(entry["tail_rank_filtered"])

        elif relation_2_type[relation] == "one to many":
            all_one_to_many_head_predictions += 1
            all_one_to_many_tail_predictions += 1
            one_to_many_head_ranks.append(entry["head_rank_filtered"])
            one_to_many_tail_ranks.append(entry["tail_rank_filtered"])

        elif relation_2_type[relation] == "many to one":
            all_many_to_one_head_predictions += 1
            all_many_to_one_tail_predictions += 1
            many_to_one_head_ranks.append(entry["head_rank_filtered"])
            many_to_one_tail_ranks.append(entry["tail_rank_filtered"])

        else:
            all_many_to_many_head_predictions += 1
            all_many_to_many_tail_predictions += 1
            many_to_many_head_ranks.append(entry["head_rank_filtered"])
            many_to_many_tail_ranks.append(entry["tail_rank_filtered"])

    print("One to one head predictions mean rank: " + str(np.average(one_to_one_head_ranks)))
    print("One to one tail predictions mean rank: " + str(np.average(one_to_one_tail_ranks)))
    print("One to many head predictions mean rank: " + str(np.average(one_to_many_head_ranks)))
    print("One to many tail predictions mean rank: " + str(np.average(one_to_many_tail_ranks)))
    print("Many to one head predictions mean rank: " + str(np.average(many_to_one_head_ranks)))
    print("Many to one tail predictions mean rank: " + str(np.average(many_to_one_tail_ranks)))
    print("Many to many head predictions mean rank: " + str(np.average(many_to_many_head_ranks)))
    print("Many to many tail predictions mean rank: " + str(np.average(many_to_many_tail_ranks)))

    all_ranks = one_to_one_head_ranks + one_to_one_tail_ranks + one_to_many_head_ranks + one_to_many_tail_ranks + many_to_one_head_ranks + many_to_one_tail_ranks + many_to_many_head_ranks + many_to_many_tail_ranks
    all_counts = all_one_to_one_head_predictions + all_one_to_one_tail_predictions + all_one_to_many_head_predictions + all_one_to_many_tail_predictions + \
        all_many_to_one_head_predictions + all_many_to_one_tail_predictions + all_many_to_many_head_predictions + all_many_to_many_tail_predictions

    overall_mean_rank = np.average(all_ranks)
    print("Overall mean rank: " + str(overall_mean_rank))

    print()


print("HITS ANALYSIS")
for model_name in model_name_2_entries:

    print(model_name)

    one_to_one_head_hits, all_one_to_one_head_predictions = 0, 0
    one_to_one_tail_hits, all_one_to_one_tail_predictions = 0, 0
    one_to_many_head_hits, all_one_to_many_head_predictions = 0, 0
    one_to_many_tail_hits, all_one_to_many_tail_predictions = 0, 0
    many_to_one_head_hits, all_many_to_one_head_predictions = 0, 0
    many_to_one_tail_hits, all_many_to_one_tail_predictions = 0, 0
    many_to_many_head_hits, all_many_to_many_head_predictions = 0, 0
    many_to_many_tail_hits, all_many_to_many_tail_predictions = 0, 0

    for entry in model_name_2_entries[model_name]:
        relation = entry['relation']

        overall = 0
        if relation_2_type[relation] == "one to one":
            all_one_to_one_head_predictions += 1
            all_one_to_one_tail_predictions += 1
            if entry["head_rank_filtered"] <= 1:
                one_to_one_head_hits += 1
            if entry["tail_rank_filtered"] <= 1:
                one_to_one_tail_hits += 1

        elif relation_2_type[relation] == "one to many":
            all_one_to_many_head_predictions += 1
            all_one_to_many_tail_predictions += 1

            if entry["head_rank_filtered"] <= 1:
                one_to_many_head_hits += 1
            if entry["tail_rank_filtered"] <= 1:
                one_to_many_tail_hits += 1

        elif relation_2_type[relation] == "many to one":
            all_many_to_one_head_predictions += 1
            all_many_to_one_tail_predictions += 1

            if entry["head_rank_filtered"] <= 1:
                many_to_one_head_hits += 1
            if entry["tail_rank_filtered"] <= 1:
                many_to_one_tail_hits += 1

        else:
            all_many_to_many_head_predictions += 1
            all_many_to_many_tail_predictions += 1

            if entry["head_rank_filtered"] <= 1:
                many_to_many_head_hits += 1
            if entry["tail_rank_filtered"] <= 1:
                many_to_many_tail_hits += 1

    print("One to one head predictions hits@1: %f%%" % (float(one_to_one_head_hits)*100/all_one_to_one_head_predictions))
    print("One to one tail predictions hits@1: %f%%" % (float(one_to_one_tail_hits)*100/all_one_to_one_tail_predictions))
    print("One to many head predictions hits@1: %f%%" % (float(one_to_many_head_hits)*100/all_one_to_many_head_predictions))
    print("One to many tail predictions hits@1: %f%%" % (float(one_to_many_tail_hits)*100/all_one_to_many_tail_predictions))
    print("Many to one head predictions hits@1: %f%%" % (float(many_to_one_head_hits)*100/all_many_to_one_head_predictions))
    print("Many to one tail predictions hits@1: %f%%" % (float(many_to_one_tail_hits)*100/all_many_to_one_tail_predictions))
    print("Many to many head predictions hits@1: %f%%" % (float(many_to_many_head_hits)*100/all_many_to_many_head_predictions))
    print("Many to many tail predictions hits@1: %f%%" % (float(many_to_many_tail_hits)*100/all_many_to_many_tail_predictions))

    hits_1_count = one_to_one_head_hits + one_to_one_tail_hits + one_to_many_head_hits + one_to_many_tail_hits + many_to_one_head_hits + many_to_one_tail_hits + many_to_many_head_hits + many_to_many_tail_hits
    all_count = all_one_to_one_head_predictions + all_one_to_one_tail_predictions + all_one_to_many_head_predictions + all_one_to_many_tail_predictions + \
        all_many_to_one_head_predictions + all_many_to_one_tail_predictions + all_many_to_many_head_predictions + all_many_to_many_tail_predictions

    overall_percentage = float(hits_1_count*100)/all_count
    print("Overall predictions hits@1: %f%%" % overall_percentage)
    print()