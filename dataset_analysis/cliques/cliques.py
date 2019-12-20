"""
    A clique is a graph in which, given any two nodes n1 and n2, there is an edge connecting them.
    In other words, each node is connected to all the others in the graph.

    In this module we identify all the cliques in the original dataset, taking into account
    the union of training set and test set.

    This allows us to compute, for each test fact, the size of the maximal clique that it belongs to.
"""
import html
import os
from collections import defaultdict

import datasets
import networkx as nx

FOLDER = "cliques"

TEST_FACTS_WITH_MAXIMAL_CLIQUE_SIZE_FILENAME = "test_facts_with_clique_size.csv"

def compute(dataset, separator=";"):
    """
    Compute, for each test triple, the size of the maximal clique that contains that triple.
    The maximal clique will be identified considering the union of training and test facts.

    :param dataset: the dataset to compute the mappings for
    :param separator: the separator to use for the keys of the result map

    :return: a map that, for each test fact <h, r, t> to the size of the maximal clique containing it
    """

    print("Computing maximal clique size for test facts of dataset %s..." % dataset.name)

    # build a non directed networkx graph containing all train_triples and test_triples
    graph = nx.Graph()
    for triple in dataset.train_triples + dataset.test_triples:
        graph.add_edge(triple[0], triple[2])

    # extract all maximal cliques in the graph
    all_cliques = list(nx.find_cliques(graph))

    # map each test triple to the size of the maximal clique it belongs to.
    # to do this, find the maximal clique containing both head and tail of the test triple:
    # that is the maximal clique containing the test fact itself.
    test_triple_2_maximal_clique_size = defaultdict(lambda: 0)
    for test_triple in dataset.test_triples:
        key = separator.join(test_triple)

        head = test_triple[0]
        tail = test_triple[2]

        for clique in all_cliques:
            if head in clique and tail in clique:
                test_triple_2_maximal_clique_size[key] = max(test_triple_2_maximal_clique_size[key], len(clique))

    return test_triple_2_maximal_clique_size

def save(dataset, write_separator=";"):
    """
    Compute, for each test triple, the size of the maximal clique that contains that triple
    and write it in a file in the local filesystem.

    :param dataset: the dataset to compute the mappings for
    :param write_separator: the separator to use when writing the mappings into the filesystem
    """

    triple_2_clique_size = compute(dataset, write_separator)

    lines = []
    for key in triple_2_clique_size:
        lines.append(key + write_separator + str(triple_2_clique_size[key]) + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_MAXIMAL_CLIQUE_SIZE_FILENAME)

    print("Saving maximal clique size for test facts of dataset %s into location %s" % (dataset.name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)


def read(dataset_name, read_separator=";", return_fact_2_clique_size=False):
    """
    Read from the filesystem a map that associates each test triple
    to the size of the maximal clique that contains that triple.

    :return return a map <clique size -> list of facts with that clique size>

    :param dataset_name: the name of the dataset to read the mappings for
    :param read_separator: the separator to use when reading the mappings from the filesystem
    :param return_fact_2_clique_size: return a map <fact -> clique size>

    """

    print("Reading number of siblings for training facts of dataset %s" % dataset_name)

    filepath = os.path.join(datasets.home_folder_for(dataset_name), FOLDER, TEST_FACTS_WITH_MAXIMAL_CLIQUE_SIZE_FILENAME)
    with open(filepath, "r") as input_file:
        lines = input_file.readlines()

    if return_fact_2_clique_size:
        triple_2_clique_size = dict()

        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, max_clique_size = line.strip().split(read_separator)

            triple_2_clique_size[read_separator.join([head, relation, tail])] = int(max_clique_size)

        return triple_2_clique_size
    else:
        clique_size_2_triples = defaultdict(lambda:[])

        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, max_clique_size = line.strip().split(read_separator)

            clique_size_2_triples[int(max_clique_size)].append(read_separator.join([head, relation, tail]))

        return clique_size_2_triples

# save(datasets.Dataset(datasets.FB15K))
