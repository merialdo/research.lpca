"""
for each relation r, compute the cardinalities as documented in compute_relation_cardinalities.py.
Then, use the cardinalities to define the class of each relationship:
    "one to one", "one to many", "many to one", "many to many"
"""
import os

import numpy

import datasets
from dataset_analysis.relation_cardinalities import relation_cardinalities

COARSE_CLASSES = ["one to one", "one to many", "many to one", "many to many"]

FILENAME = "relation_coarse_classes.csv"

def compute(dataset):
    """
    Compute the mappings  < relation coarse class -> relations that belong to that class> in a specific dataset

    :param dataset: the dataset to compute the mappings for
    :return: a dict that associates each relation coarse class class to the corresponding relations
    """

    print("Computing coarse-grained relation classes for dataset %s" % dataset.name)

    #this is a dict of dicts
    # for each relation rel it contains a dict() with two keys: "head_to_tails" and "tail_to_heads"
    # relation_2_matches[rel]["head_to_tails"] is, in turn, a dict
    #       that associates each head to the tails that it is connected to via rel
    rel_2_cardinality_dicts = relation_cardinalities.compute(dataset)

    one_2_one = []
    one_2_many = []
    many_2_one = []
    many_2_many = []

    for rel in rel_2_cardinality_dicts:
        head_2_tails = rel_2_cardinality_dicts[rel]["head_to_tails"]
        avg_tails_for_head = numpy.average([len(head_2_tails[head]) for head in head_2_tails])

        tail_2_heads = rel_2_cardinality_dicts[rel]["tail_to_heads"]
        avg_heads_for_tail = numpy.average([len(tail_2_heads[tail]) for tail in tail_2_heads])

        if avg_tails_for_head <= 1.5 and avg_heads_for_tail <= 1.5:
            one_2_one.append(rel)
        elif avg_tails_for_head <= 1.5 and avg_heads_for_tail > 1.5:
            many_2_one.append(rel)
        elif avg_tails_for_head > 1.5 and avg_heads_for_tail <= 1.5:
            one_2_many.append(rel)
        else:
            many_2_many.append(rel)

    result = dict()

    result["one to one"] = one_2_one
    result["many to one"] = many_2_one
    result["one to many"] = one_2_many
    result["many to many"] = many_2_many
    return result


def save(dataset):
    """
    Compute the mappings  < relation coarse class -> relations that belong to that class> in a specific dataset
    and save them in a location in the home folder of the dataset

    :param dataset: the dataset to compute the mappings for
    """

    coarse_class_2_rels = compute(dataset)

    lines = []
    for coarse_class in COARSE_CLASSES:
        for rel in coarse_class_2_rels[coarse_class]:
            lines.append(";".join([rel, coarse_class]) + "\n")


    output_filepath = os.path.join(dataset.home, FILENAME)

    print("Saving coarse-grained relation classes for dataset %s into location %s" % (dataset.name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)


def read(dataset_name, read_separator=";", return_rel_2_class=False):
    """
    Read the mappings <relation coarse class -> relations belonging to that coarse class >
    from the corresponding file of the dataset with the given name
    and return them
        - either in the format  <relation coarse class -> relations belonging to that coarse class >
        - or in the format  <relation -> relation coarse class that it belongs to >

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :param return_rel_2_class: if true, return mappings in the format <relation -> class that it belongs to >
                                otherwise, return mappings in the format in the format < class -> relations belonging to that class >

    :return: the computed mappings
    """

    print("Reading coarse-grained relation classes for dataset %s" % dataset_name)

    dataset_home = datasets.home_folder_for(dataset_name)

    with open(os.path.join(dataset_home, FILENAME), "r") as input_file:

        if not return_rel_2_class:
            coarse_class_2_rels = dict()
            for coarse_class in COARSE_CLASSES:
                coarse_class_2_rels[coarse_class] = []

            for line in input_file.readlines():
                relation, coarse_class = line.strip().split(read_separator)
                coarse_class_2_rels[coarse_class].append(relation)
            return coarse_class_2_rels

        else:
            rel_2_class = dict()
            for line in input_file.readlines():
                relation, coarse_class = line.strip().split(read_separator)
                rel_2_class[relation] = coarse_class
            return rel_2_class

# dataset = datasets.Dataset(datasets.FB15K)
# save(dataset)