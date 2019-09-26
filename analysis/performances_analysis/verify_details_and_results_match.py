import os

import numpy
# read entities and create a map mid -> id
import datasets
import models
import performances
from dataset_analysis.degrees import entity_degrees
from datasets import FB15K
from models import TRANSE

prediction_2_rank = dict()
prediction_2_mids = dict()


def verify_metrics_for(model_name, dataset_name):
    print("Verifying that filtered ranks and details match for " + model_name + " on " + dataset_name + "...")

    entity_2_in_degree, entity_2_out_degree, entity_2_degree = entity_degrees.read(dataset_name)

    num_of_entities = len(entity_2_degree)

    entries_1 = performances.read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=num_of_entities)
    entries_2 = performances.read_filtered_details_entries_for(model_name, dataset_name, entity_number=num_of_entities)

    e1_key_2_ranks = dict()
    e2_key_2_ranks = dict()

    for e1 in entries_1:
        key = ";".join([e1["head"], e1["relation"], e1["tail"]])
        e1_key_2_ranks[key] = (e1["head_rank_filtered"], e1["tail_rank_filtered"])

    for e2 in entries_2:
        key = ";".join([e2["head"], e2["relation"], e2["tail"]])
        e2_key_2_ranks[key] = (e2["head_rank_filtered"], e2["tail_rank_filtered"])


    for key in e1_key_2_ranks:
        if e1_key_2_ranks[key] != e2_key_2_ranks[key]:
            print(key)
            print(e1_key_2_ranks[key])
            print(e2_key_2_ranks[key])



for cur_dataset_name in datasets.ALL_DATASET_NAMES:
    for cur_model_name in [models.COMPLEX]:    # models.ALL_NAMES:
        filtered_ranks_path = models.filtered_ranks_path(cur_model_name, cur_dataset_name)
        filtered_details_path = models.filtered_details_path(cur_model_name, cur_dataset_name)

        if not os.path.isfile(filtered_ranks_path) or not os.path.isfile(filtered_details_path):
            continue

        verify_metrics_for(cur_model_name, cur_dataset_name)
        print("Done\n")




