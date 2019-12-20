import html
from collections import defaultdict
import os
import datasets

FOLDER = "degrees"
FILENAME = "relation_mentions.csv"

def compute(dataset):
    """
    Compute the mapping <relation name -> number of mentions>
        from the training set of a specific dataset.

    :param dataset: the dataset to compute the mappings for
    :return: the computed mappings

    """

    print("Computing the mapping <relation name -> number of mentions> in %s training set..." % dataset.name)
    relation_2_mentions = defaultdict(lambda: 0)

    for head, relation, tail in dataset.train_triples:
        relation_2_mentions[relation] += 1

    return relation_2_mentions


def read(dataset_name, read_separator=";"):
    """
    Read the file that contains the mapping <relation name -> number of mentions>
    for the training set of a specific dataset.
    :param dataset_name: the name of the dataset to read the mappings for, from its specific, pre-computed file
    :param read_separator: the separator to use when reading the file
    :return: the read mappings
    """

    print("Reading the mapping <relation name -> number of mentions> in %s training set..." % dataset_name)
    dataset_home = datasets.home_folder_for(dataset_name)
    filepath = os.path.join(dataset_home, FOLDER, FILENAME)

    name_2_count = defaultdict(lambda: 0)
    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            (name, count) = line.strip().split(read_separator)
            name_2_count[name] = int(count)

    return name_2_count


def save(dataset_name, write_separator=";"):
    """
    Compute the mapping <relation name -> number of mentions>
        from the training set of a specific dataset
        and save it in a specific file in the home folder for that dataset
    :param dataset_name: the name of the dataset to compute and save the mappings for
    :param write_separator: the separator to use when writing the mappings
    """

    dataset = datasets.Dataset(dataset_name)
    relation_2_mentions = compute(dataset)

    lines = []
    for relation in relation_2_mentions:
        lines.append(write_separator.join([relation, str(relation_2_mentions[relation])]) + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, FILENAME)

    print("Writing the mapping <relation name -> number of mentions> for %s training set in %s..." % (dataset_name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)

#save(datasets.FB15K)
#save(datasets.WN18)
#save(datasets.FB15K_237)
#save(datasets.WN18RR)
# save(datasets.YAGO3_10)