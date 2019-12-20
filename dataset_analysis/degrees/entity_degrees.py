import html
import os
from collections import defaultdict

import datasets

FOLDER = "degrees"
FILENAME = "entity_degrees.csv"

def compute(dataset):
    """
    Compute the mappings
            <entity name -> in degree>
            <entity name -> out degree>
            <entity name -> overall degree>
        from the training set of a specific dataset.

    :param dataset: the dataset to compute the mappings for
    :return: the computed mappings, in the order <entity -> in degree>, <entity -> out degree>, <entity -> overall degree>
    """

    print("Computing the mappings <entity name -> degree> (for in, out and overall degree) in %s training set..." % dataset.name)

    entity_2_in_degree = defaultdict(lambda: 0)
    entity_2_out_degree = defaultdict(lambda: 0)
    entity_2_degree = defaultdict(lambda: 0)

    for (head, relation, tail) in dataset.train_triples:
        entity_2_out_degree[head] += 1
        entity_2_in_degree[tail] += 1
        entity_2_degree[head] += 1
        entity_2_degree[tail] += 1

    return entity_2_in_degree, entity_2_out_degree, entity_2_degree


def read(dataset_name, read_separator=";"):
    """

    Read the file that contains the mappings
            <entity name -> in degree>
            <entity name -> out degree>
            <entity name -> overall degree>
    for the training set of a specific dataset.

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :return: the computed mappings, in the order <entity -> in degree>, <entity -> out degree>, <entity -> overall degree>
    """

    print("Reading the mappings <entity name -> degree> (for in, out and overall degree) in %s training set..." % dataset_name)
    dataset_home = datasets.home_folder_for(dataset_name)
    filepath = os.path.join(dataset_home, FOLDER, FILENAME)

    mid_2_in_degree = defaultdict(lambda: 0)
    mid_2_out_degree = defaultdict(lambda: 0)
    mid_2_degree = defaultdict(lambda: 0)

    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            (mid, in_degree, out_degree, degree) = line.strip().split(read_separator)
            mid_2_in_degree[mid] = int(in_degree)
            mid_2_out_degree[mid] = int(out_degree)
            mid_2_degree[mid] = int(degree)

    return mid_2_in_degree, mid_2_out_degree, mid_2_degree


def save(dataset_name, write_separator = ";"):
    """
    Compute the mappings
            <entity name -> in degree>
            <entity name -> out degree>
            <entity name -> overall degree>
    and save them in a file.

    :param write_separator: the separator to use when writing the file
    :param dataset_name: the name of the dataset for which to compute the mappings
    """

    dataset = datasets.Dataset(dataset_name)
    entity_in_degrees, entity_out_degrees, entity_degrees = compute(dataset)

    lines = []
    for entity in entity_degrees:
        lines.append(write_separator.join([entity,
                                     str(entity_in_degrees[entity]),
                                     str(entity_out_degrees[entity]),
                                     str(entity_degrees[entity])]) + "\n")


    output_filepath = os.path.join(dataset.home, FOLDER, FILENAME)

    print("Writing the mappings <entity name -> in, out and overall degree> for %s training set in %s..." % (dataset_name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)