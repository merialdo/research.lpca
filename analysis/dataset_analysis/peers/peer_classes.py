"""
for each fact in test set, get the count the head peers and tail peers seen in training set
Then, use these peers to to define the fine grained class of each fact.
"""
import html
import os
from collections import defaultdict

import datasets
from dataset_analysis.peers import peers

FOLDER = "peers"
TEST_FACTS_WITH_PEERS_FILENAME = "test_facts_with_peer_classes.csv"

# 9 intervals:
#   [0, 1)
#   [1, 2)
#   [2, 4)
#   [4, 8)
#   [8, 16)
#   [16, 32)
#   [32, 64)
#   [64, 128)
#   [128, inf)
PEER_INTERVALS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, "inf")]

PEER_CLASSES = []
PEER_CLASS_2_INTERVALS = dict()

# use all possible intervals for both head and tail peers
for head_peers_interval in PEER_INTERVALS:
    for tail_peers_interval in PEER_INTERVALS:
        peer_class_name = str(head_peers_interval[0]) + "-" + str(head_peers_interval[1]) + "__" + str(tail_peers_interval[0]) + "-" + str(tail_peers_interval[1])
        PEER_CLASSES.append(peer_class_name)
        PEER_CLASS_2_INTERVALS[peer_class_name] = (head_peers_interval, tail_peers_interval)


def _fact_belongs_to_peer_class(head_peers, tail_peers, class_name):

    if class_name not in PEER_CLASSES:
        return False

    (head_peers_interval, tail_peers_interval) = PEER_CLASS_2_INTERVALS[class_name]

    head_peers_lower_bound = head_peers_interval[0]
    head_peers_upper_bound = 10**9 if head_peers_interval[1] == 'inf' else head_peers_interval[1]

    tail_peers_lower_bound = tail_peers_interval[0]
    tail_peers_upper_bound = 10**9 if tail_peers_interval[1] == 'inf' else tail_peers_interval[1]

    return (head_peers_lower_bound <= head_peers < head_peers_upper_bound and
            tail_peers_lower_bound <= tail_peers < tail_peers_upper_bound)



def compute(dataset):
    """
    Compute the mappings  < peer class -> test facts that belong to that class >   in a specific dataset

    :param dataset: the dataset to compute the mappings for
    :return: a dict that associates each peer class class to the corresponding test facts
    """

    print("Computing peer classes for test facts of dataset %s" % dataset.name)

    # read the peers counts (computed on training facts)
    # in the form of "head question -> head answers" (that are the head peers)
    # and "tail question -> tail answers" (that are the tail peers)
    # NOTE: THESE ARE DEFAULTDICTS, SO IN CASE A QUESTION IS UNKNOWN, THE ANSWERS SIBLINGS ARE 0 (CORRECTLY)
    head_prediction_2_peers, tail_prediction_2_peers = peers.read(dataset.name)

    # compute
    peer_class_2_facts = defaultdict(lambda:[])

    # for each test fact, get the head peers and tail peers
    for (head, relation, tail) in dataset.test_triples:

        head_peers = head_prediction_2_peers[relation + ";" + tail]
        tail_peers = tail_prediction_2_peers[head + ";" + relation]

        for peer_class in PEER_CLASSES:
            if _fact_belongs_to_peer_class(head_peers, tail_peers, peer_class):
                peer_class_2_facts[peer_class].append((head, relation, tail))
                break

    return peer_class_2_facts


def read(dataset_name, read_separator=";", return_fact_2_class=False):
    """
    Read the mappings <peer class -> test facts to that class >
    from the corresponding file of the dataset with the given name
    and return them
        - either in the format  <peer class -> test facts belonging to that class >
        - or in the format  <test fact -> peer class that it belongs to >

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :param return_fact_2_class: if true, return mappings in the format <test fact -> peer class that it belongs to >
                                otherwise, return mappings in the format in the format <peer class -> test facts belonging to that class >

    :return: the computed mappings
    """

    print("Reading peer classes for test facts of dataset %s" % dataset_name)

    input_filepath = os.path.join(datasets.home_folder_for(dataset_name), FOLDER, TEST_FACTS_WITH_PEERS_FILENAME)

    with open(input_filepath, "r") as input_file:
        if not return_fact_2_class:
            peer_class_2_facts = dict()
            for peer_class in PEER_CLASSES:
                peer_class_2_facts[peer_class] = []

            for line in input_file.readlines():
                line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
                head, relation, tail, peer_class = line.strip().split(read_separator)
                peer_class_2_facts[peer_class].append([head, relation, tail])
            return peer_class_2_facts

        else:
            fact_2_peer_class = dict()
            for line in input_file.readlines():
                line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
                head, relation, tail, peer_class = line.strip().split(read_separator)
                fact_2_peer_class[";".join([head, relation, tail])] = peer_class
            return fact_2_peer_class


def save(dataset):
    """
    Compute the mappings  < peer class -> test facts that belong to that class> in a specific dataset
    and save them in a location in the home folder of the dataset

    :param dataset: the dataset to compute the mappings for
    """

    peer_class_2_facts = compute(dataset)
    lines = []
    for peer_class in PEER_CLASSES:
        for (head, relationship, tail) in peer_class_2_facts[peer_class]:
            lines.append(";".join([head, relationship, tail, peer_class]) + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_PEERS_FILENAME)
    print("Saving peer classes for test facts of dataset %s into location %s" % (dataset.name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)


#save(datasets.Dataset(datasets.FB15K))
#save(datasets.Dataset(datasets.FB15K_237))
#save(datasets.Dataset(datasets.WN18))
#save(datasets.Dataset(datasets.WN18RR))
#save(datasets.Dataset(datasets.YAGO3_10))
