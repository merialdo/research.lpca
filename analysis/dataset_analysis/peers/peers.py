"""
This is an utility module to compute peers.

Given a fact in a dataset:
    - its "head peers" are all the entities that are valid alternatives for the fact head.
    - its "tail peers" are all the entities that are valid alternatives for the fact tail.

In other words, given any fact <h, r, t>,
    - if other facts { <k, r, t>,  <j, r, t>,  ... } exist in the training set,
        h, k, and j are head peers for <_, r, t> because they are all valid heads for the same relation r and tail t.

    - similarly, if other facts { <h, r, s>,  <h, r, q>,  ... } exist in the training set,
        s, t, and q are tail peers for <h, r, _> because they are all valid tails for the same head h amd relation r.

"""
import html
import os
from collections import defaultdict

import datasets

FOLDER = "peers"
TRAIN_FACTS_WITH_PEERS_FILENAME = "train_facts_with_peer_numbers.csv"

def compute(dataset):
    """
    Compute the number of head peers and tail peers for each training fact of a specific dataset.
    The number of peers will be computed taking into account only the training set.

    :param dataset: the dataset to compute the mappings for
    """

    print("Computing number of peers for training facts of dataset %s" % dataset.name)

    # match each <?, r, t> "question" to the count of entities h that answer correctly that "question"
    head_question_2_answers_count = defaultdict(lambda:0)
    # match each <h, r, ?> "question" to the count of entities t that answer correctly that "question"
    tail_question_2_answers_count = defaultdict(lambda:0)
    for (head, rel, tail) in dataset.train_triples:
        head_question_2_answers_count[rel + ";" + tail] += 1    #prediction to heads
        tail_question_2_answers_count[head + ";" + rel] += 1    #prediction to tails

    return head_question_2_answers_count, tail_question_2_answers_count

def save(dataset, write_separator=";"):
    """
    Compute for the training facts of a dataset
    the number of head peers for all head entities and the number of tail peers for all tail entities
    and write it in a file in the local filesystem.

    :param dataset: the dataset to compute the mappings for
    :param write_separator: the separator to use when writing the mappings into the filesystem
    """

    # head question vs number of answers (all the answers for a same head questions are head peers)
    head_question_2_answers_count, tail_question_2_answers_count = compute(dataset)

    lines = []
    for (head, relation, tail) in dataset.train_triples:
        head_peers = head_question_2_answers_count[relation + ";" + tail]
        tail_peers = tail_question_2_answers_count[head + ";" + relation]
        lines.append(write_separator.join([head, relation, tail, str(head_peers), str(tail_peers)]) + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_PEERS_FILENAME)

    print("Saving number of peers for training facts of dataset %s into location %s" % (dataset.name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)


def read(dataset_name, read_separator=";"):
    """
    Read the mappings <prediction -> number of correct answers >
    from the corresponding file of the dataset with the given name
    and return them in the format of two dicts:
        - head_prediction_2_peers: maps all "head questions" relation;tail_entity to the number of correct head answers
        - tail_prediction_2_peers: maps all "tail questions" head_entity;relation to the number of correct tail answers

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file

    :return: the computed mappings
    """

    print("Reading number of peers for training facts of dataset %s" % dataset_name)

    filepath = os.path.join(datasets.home_folder_for(dataset_name), FOLDER, TRAIN_FACTS_WITH_PEERS_FILENAME)
    head_prediction_2_peers = defaultdict(lambda:0)
    tail_prediction_2_peers = defaultdict(lambda:0)

    with open(filepath, "r") as input_file:
        lines = input_file.readlines()
        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, head_peers, tail_peers = line.strip().split(read_separator)

            head_prediction_2_peers[relation + ";" + tail] = int(head_peers)
            tail_prediction_2_peers[head + ";" + relation] = int(tail_peers)

    return head_prediction_2_peers, tail_prediction_2_peers

#save(datasets.Dataset(datasets.FB15K))
#save(datasets.Dataset(datasets.FB15K_237))
#save(datasets.Dataset(datasets.WN18))
#save(datasets.Dataset(datasets.WN18RR))
#save(datasets.Dataset(datasets.YAGO3_10))
