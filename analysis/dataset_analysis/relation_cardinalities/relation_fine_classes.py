"""
for each relation r, compute the cardinalities as documented in compute_relation_cardinalities.py.
Then, use the cardinalities to define the fine grained class class of each relationship.
"""
import os
from collections import defaultdict

import datasets


FILENAME = "relation_fine_classes.csv"

INTERVALS = [(0, 1.5), (1.5, 4), (4, 25), (25, 100), (100, "inf")]
FINE_CLASSES = []
FINE_CLASS_2_INTERVALS = dict()

for heads_per_tail_interval  in INTERVALS:
    for tails_per_head_interval in INTERVALS:
        class_name = str(tails_per_head_interval[0]) + "-" + str(tails_per_head_interval[1]) + "__" + str(heads_per_tail_interval[0]) + "-" + str(heads_per_tail_interval[1])
        FINE_CLASSES.append(class_name)
        FINE_CLASS_2_INTERVALS[class_name] = (heads_per_tail_interval, tails_per_head_interval)

import numpy

from dataset_analysis.relation_cardinalities import relation_cardinalities


def _relation_belongs_to_fine_class(rel_avg_heads_for_tail, rel_avg_tails_for_head, class_name):

    if class_name not in FINE_CLASSES:
        return False

    (class_tails_per_head_interval, class_heads_per_tail_interval) = FINE_CLASS_2_INTERVALS[class_name]

    heads_per_tail_lower_bound = class_heads_per_tail_interval[0]
    heads_per_tail_upper_bound = 10000000 if class_heads_per_tail_interval[1] == 'inf' else class_heads_per_tail_interval[1]

    tails_per_head_lower_bound = class_tails_per_head_interval[0]
    tails_per_head_upper_bound = 10000000 if class_tails_per_head_interval[1] == 'inf' else class_tails_per_head_interval[1]

    return (heads_per_tail_lower_bound < rel_avg_heads_for_tail <= heads_per_tail_upper_bound and
            tails_per_head_lower_bound < rel_avg_tails_for_head <= tails_per_head_upper_bound)



def compute(dataset):
    """
    Compute the mappings  < relation fine class -> relations that belong to that class> in a specific dataset

    :param dataset: the dataset to compute the mappings for
    :return: a dict that associates each relation fine class class to the corresponding relations
    """

    print("Computing fine-grained relation classes for dataset %s" % dataset.name)

    #this is a dict of dicts
    # for each relation rel it contains a dict() with two keys: "head_to_tails" and "tail_to_heads"
    # relation_2_matches[rel]["head_to_tails"] is, in turn, a dict
    #       that associates each head to the tails that it is connected to via rel
    rel_2_cardinality_dicts = relation_cardinalities.compute(dataset)

    fine_class_2_rels = defaultdict(lambda:[])

    for rel in rel_2_cardinality_dicts:
        tail_2_heads = rel_2_cardinality_dicts[rel]["tail_to_heads"]
        avg_heads_for_tail = numpy.average([len(tail_2_heads[tail]) for tail in tail_2_heads])

        head_2_tails = rel_2_cardinality_dicts[rel]["head_to_tails"]
        avg_tails_for_head = numpy.average([len(head_2_tails[head]) for head in head_2_tails])


        for fine_class in FINE_CLASSES:
            if _relation_belongs_to_fine_class(avg_heads_for_tail, avg_tails_for_head, fine_class):
                fine_class_2_rels[fine_class].append(rel)

    return fine_class_2_rels


def read(dataset_name, read_separator=";", return_rel_2_class=False):
    """
    Read the mappings <relation fine class -> relations belonging to that fine class >
    from the corresponding file of the dataset with the given name
    and return them
        - either in the format  <relation fine class -> relations belonging to that fine class >
        - or in the format  <relation -> relation fine class that it belongs to >

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :param return_rel_2_class: if true, return mappings in the format <relation -> class that it belongs to >
                                otherwise, return mappings in the format in the format < class -> relations belonging to that class >

    :return: the computed mappings
    """

    print("Reading fine-grained relation classes for dataset %s" % dataset_name)

    dataset_home = datasets.home_folder_for(dataset_name)

    with open(os.path.join(dataset_home, FILENAME), "r") as input_file:

        if not return_rel_2_class:
            fine_class_2_rels = dict()
            for fine_class in FINE_CLASSES:
                fine_class_2_rels[fine_class] = []

            for line in input_file.readlines():
                relation, fine_class = line.strip().split(read_separator)
                fine_class_2_rels[fine_class].append(relation)
            return fine_class_2_rels

        else:
            rel_2_class = dict()
            for line in input_file.readlines():
                relation, fine_class = line.strip().split(read_separator)
                rel_2_class[relation] = fine_class
            return rel_2_class


def save(dataset):
    """
    Compute the mappings  < relation fine class -> relations that belong to that class> in a specific dataset
    and save them in a location in the home folder of the dataset

    :param dataset: the dataset to compute the mappings for
    """

    fine_class_2_rels = compute(dataset)
    lines = []
    for fine_class in FINE_CLASSES:
        for rel in fine_class_2_rels[fine_class]:
            lines.append(";".join([rel, fine_class]) + "\n")

    dataset_home = datasets.home_folder_for(dataset.home)

    output_filepath = os.path.join(dataset_home, FILENAME)

    print("Saving fine-grained relation classes for dataset %s into location %s" % (dataset.name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)

# dataset = datasets.Dataset(datasets.FB15K)
# save(dataset)