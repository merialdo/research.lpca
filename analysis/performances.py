import html
import os
from collections import defaultdict

from dataset_analysis.degrees import entity_degrees

HOME = "/Users/andrea/comparative_analysis/results"

FILTERED_RANKS_ENTRIES_SUFFIX = "_filtered_ranks.csv"
FILTERED_DETAILS_ENTRIES_SUFFIX = "_filtered_details.csv"

def read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=None):
    print("Reading filtered rank entries for " + model_name + " results on " + dataset_name + "...")
    filepath = os.path.join(HOME, dataset_name, model_name + FILTERED_RANKS_ENTRIES_SUFFIX)

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
                entry["head_rank_filtered"] = float(entity_number)
            else:
                entry["head_rank_filtered"] = float(rank_head_filtered)

            if rank_tail_filtered.startswith("MISS_"):
                entry["tail_rank_filtered"] = float(entity_number)
            else:
                entry["tail_rank_filtered"] = float(rank_tail_filtered)

            entries.append(entry)

    return entries

def read_filtered_details_entries_for(model_name, dataset_name, entity_number=None):
    print("Reading filtered details entries for " + model_name + " results on " + dataset_name + "...")

    filepath = os.path.join(HOME, dataset_name, model_name + FILTERED_DETAILS_ENTRIES_SUFFIX)

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
            entry["head_rank_filtered"] = float(entity_number)
        else:
            entry["head_rank_filtered"] = float(len(head_details))

        if tail_details[-1].startswith("MISS_"):
            entry["tail_rank_filtered"] = float(entity_number)
        else:
            entry["tail_rank_filtered"] = float(len(tail_details))


        entries.append(entry)
    return entries