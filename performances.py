import html
import os
from collections import defaultdict

from config import RESULTS_HOME
from dataset_analysis.degrees import entity_degrees
from models import ANYBURL, ALL_MODEL_NAMES

FILTERED_RANKS_ENTRIES_SUFFIX = "_filtered_ranks"
FILTERED_DETAILS_ENTRIES_SUFFIX = "_filtered_details"
EXTENSION = ".csv"
def read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=None, tie_policy="average"):
    print("Reading filtered rank entries for " + model_name + " results on " + dataset_name + " with tie policy " + tie_policy + "...")

    filepath = filtered_ranks_path(model_name, dataset_name, tie_policy)

    if entity_number is None:
        _, _, entity_2_degree = entity_degrees.read(dataset_name)
        entity_number = len(entity_2_degree)

    entries = []

    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:

            line = html.unescape(line)
            (head, relation, tail,
             rank_head_filtered,
             rank_tail_filtered) = line.strip().split(";")

            entry = dict()
            entry["head"] = head
            entry["relation"] = relation
            entry["tail"] = tail

            if rank_head_filtered.startswith("MISS_"):
                if tie_policy == "max":
                    entry["head_rank_filtered"] = float(entity_number)
                elif tie_policy == "average":
                    entry["head_rank_filtered"] = (int(rank_head_filtered.replace("MISS_", "")) + float(entity_number) ) / 2
                elif tie_policy == "min":
                    entry["head_rank_filtered"] = float(rank_head_filtered.replace("MISS_", ""))
            else:
                entry["head_rank_filtered"] = float(rank_head_filtered)

            if rank_tail_filtered.startswith("MISS_"):
                if tie_policy == "max":
                    entry["tail_rank_filtered"] = float(entity_number)
                elif tie_policy == "average":
                    entry["tail_rank_filtered"] = (int(rank_tail_filtered.replace("MISS_", "")) + float(entity_number) ) / 2
                elif tie_policy == "min":
                    entry["tail_rank_filtered"] = float(rank_tail_filtered.replace("MISS_", ""))
            else:
                entry["tail_rank_filtered"] = float(rank_tail_filtered)

            entries.append(entry)

    return entries

def read_filtered_details_entries_for(model_name, dataset_name, entity_number=None, tie_policy="average"):
    print("Reading filtered details entries for " + model_name + " results on " + dataset_name + " with tie policy " + tie_policy + "...")

    filepath = filtered_details_path(model_name, dataset_name, tie_policy)

    if entity_number is None:
        _, _, entity_2_degree = entity_degrees.read(dataset_name)
        entity_number = len(entity_2_degree)

    fact_2_head_tail_details = defaultdict(lambda : dict())

    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:

            line = html.unescape(line)
            (head, relation, tail, type, details) = line.strip().split(";", 4)
            details = details[1:-1].split(";")

            key = ";".join([head, relation, tail])
            fact_2_head_tail_details[key][type] = details


    entries = []
    for key in fact_2_head_tail_details:

        entry = dict()

        head, relation, tail = key.split(";")

        entry["head"] = head
        entry["relation"] = relation
        entry["tail"] = tail
        head_details = fact_2_head_tail_details[key]["predict head"]
        tail_details = fact_2_head_tail_details[key]["predict tail"]

        entry["head_details_filtered"] = head_details
        entry["tail_details_filtered"] = tail_details

        if head_details[-1].startswith("MISS_"):
            if tie_policy == "max":
                entry["head_rank_filtered"] = float(entity_number)
            elif tie_policy == "average":
                entry["head_rank_filtered"] = (float(len(head_details)) + float(entity_number)) / 2
            elif tie_policy == "min":
                entry["head_rank_filtered"] = float(len(head_details))
        else:
            entry["head_rank_filtered"] = float(len(head_details))

        if tail_details[-1].startswith("MISS_"):
            if tie_policy == "max":
                entry["tail_rank_filtered"] = float(entity_number)
            elif tie_policy == "average":
                entry["tail_rank_filtered"] = (float(len(tail_details)) + float(entity_number)) / 2
            elif tie_policy == "min":
                entry["tail_rank_filtered"] = float(len(tail_details))
        else:
            entry["tail_rank_filtered"] = float(len(tail_details))

        entries.append(entry)
    return entries

def filtered_ranks_path(model_name, dataset_name, tie_policy="average"):
    filepath = os.path.join(RESULTS_HOME, dataset_name, model_name, model_name + FILTERED_RANKS_ENTRIES_SUFFIX + EXTENSION)
    if model_name != ANYBURL and tie_policy == "min":
        filepath = os.path.join(RESULTS_HOME, dataset_name, model_name, model_name + FILTERED_RANKS_ENTRIES_SUFFIX + "_min" + EXTENSION)
    return os.path.abspath(filepath)

def filtered_details_path(model_name, dataset_name, tie_policy="average"):
    filepath = os.path.join(RESULTS_HOME, dataset_name, model_name, model_name + FILTERED_DETAILS_ENTRIES_SUFFIX + EXTENSION)
    if model_name != ANYBURL and tie_policy == "min":
        filepath = os.path.join(RESULTS_HOME, dataset_name, model_name, model_name + FILTERED_DETAILS_ENTRIES_SUFFIX + "_min" + EXTENSION)

    return os.path.abspath(filepath)

def get_models_supporting_dataset(dataset_name):
    result = []
    for model_name in ALL_MODEL_NAMES:
        if os.path.isfile(filtered_ranks_path(model_name, dataset_name)):
            result.append(model_name)
    return result