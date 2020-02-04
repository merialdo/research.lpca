"""
    This is an utility module to compute, for each relation type in a dataset,
    its co-occurrences with RELATION PATHS of 1 step, 2 steps and 3 steps in the training set.

    A RELATION PATH is just a sequence of relation types.

        For instance, given the subgraph

            |---place of birth---> Honolulu ----located in----|
            |                                                 |
            |                                                 |
        Barack Obama -------------nationality--------------> USA


    This is a GRAPH PATH:
        Barack Obama ---place of birth---> Honolulu ----located in----> USA

    This is a RELATION PATH:
         place of birth, located in


    A relation r co-occurs with a RELATION PATH <s, t> when in the dataset
    there is at least one fact <A, r, Z> with a GRAPH PATH <A, s, *>, <*, t, Z>.
    The FREQUENCY of a RELATION PATH <s, t> to a relation r is the number of times that,
    r co-occurs with at least one instance of that RELATION PATH.

    In our analysis, in order to compute RELATION PATHS:
        - for each fact <h, r, t> in the dataset:
            - first, we compute all the training 1-step GRAPH PATHS and training 2-step GRAPH PATHS
            - then, we obtain from these GRAPH PATHS the corresponding RELATION PATHS.
              (if the same RELATION PATH occurs multiple times under the same fact, it still counts as 1)
        - finally, we aggregate by the type of the relation r in the original fact,
        and see how many times each RELATION PATH co-occurs with r
"""

import html
import operator
import os
from collections import defaultdict

import datasets
from dataset_analysis.paths import graph_paths

FOLDER = "paths"
ONE_STEP_RELATION_PATHS_FILENAME = "relations_with_train_one_step_relation_paths_frequency.csv"
TWO_STEP_RELATION_PATHS_FILENAME = "relations_with_train_two_step_relation_paths_frequency.csv"
THREE_STEP_RELATION_PATHS_FILENAME = "relations_with_train_three_step_relation_paths_frequency.csv"

RELATION_PATHS_IN_LIST_SEPARATOR = "|"
SEPARATOR = ";"

def _build_key_for(ent1, ent2):
    """
        Utility method for building a string dict key based on the names of two entities.
        This method computes THE SAME KEY for both (ent1, ent2) and (ent2, ent1).

        :param ent1: the first entity in the pair of entities to build the key for
        :param ent2: the second entity in the pair of entities to build the key for
        :return: the built key
    """
    return SEPARATOR.join(sorted([ent1, ent2]))


def compute(dataset, max_steps=3):
    """
        Compute for a specific dataset all the training RELATION PATHS up to a specific maximum length (between 1 and 3)
        and their frequencies to each relation type in the training set.

        To do so, only the training set of the dataset is taken into account.

        :param dataset: the dataset to compute the graph paths for
        :param max_steps: the maximum number of steps per path to take into account.
                           The highest value allowed for max_steps is currently 3

        :return: two dictionaries containing, for each relation in training set,
                    a dictionary that maps each relation path to its frequency to that relation.
                 The two dictionaries refer to 1-step and 2-step relation paths respectively
    """

    if max_steps <= 0:
        raise Exception("Invalid value for \"max_steps\": " + str(max_steps))
    if max_steps > 3:
        raise Exception("The maximum value supported for \"max_steps\" is 3")

    # these dicts map each relation type in the training set to a nested dict
    #   that maps each relation path (1-step or 2-step path respectively) to its frequency to the relation
    relation_2_one_step_relation_paths = defaultdict(lambda:defaultdict(lambda: 0))
    relation_2_two_step_relation_paths = defaultdict(lambda:defaultdict(lambda: 0))
    relation_2_three_step_relation_paths = defaultdict(lambda:defaultdict(lambda: 0))

    if max_steps == 1:
        train_fact_to_one_step_graph_paths = graph_paths.read_train(dataset, max_steps=max_steps)
    elif max_steps == 2:
        train_fact_to_one_step_graph_paths, train_fact_to_two_step_graph_paths = graph_paths.read_train(dataset, max_steps=max_steps)
    else:
        train_fact_to_one_step_graph_paths, train_fact_to_two_step_graph_paths, train_fact_to_three_step_graph_paths = graph_paths.read_train(dataset, max_steps=max_steps)

    print("Computing frequencies of 1-step relation paths for dataset %s..." % dataset.name)

    # for each training fact (that MAY be a self-loop)
    for fact_key in train_fact_to_one_step_graph_paths:

        # get its relation
        _, fact_relation, _ = fact_key.split(SEPARATOR)

        # get all the 1-step GRAPH PATHS from the fact head to the fact tail
        one_step_graph_paths = train_fact_to_one_step_graph_paths[fact_key]
        # extract the set of 1-step RELATION PATHS from the 1-step GRAPH PATHS,
        # and for each one of them add 1 to the result dict under the fact_relation
        relation_paths_encountered_for_this_fact = set()
        for one_step_graph_path in one_step_graph_paths:
            (_, one_step_relation_path, _) = one_step_graph_path
            relation_paths_encountered_for_this_fact.add(one_step_relation_path)

        for one_step_relation_path in relation_paths_encountered_for_this_fact:
            relation_2_one_step_relation_paths[fact_relation][one_step_relation_path] += 1

    if max_steps == 1:
        return relation_2_one_step_relation_paths

    print("Computing frequencies of 2-step relation paths for dataset %s..." % dataset.name)
    # for each training fact (that MAY be a self-loop)
    for fact_key in train_fact_to_two_step_graph_paths:

        # get its relation
        _, fact_relation, _ = fact_key.split(SEPARATOR)

        # get all the 2-step GRAPH PATHS from the fact head to the fact tail
        two_step_graph_paths = train_fact_to_two_step_graph_paths[fact_key]

        # extract the set of 2-step RELATION PATHS from the 2-step GRAPH PATHS,
        # and for each one of them add 1 to the result dict under the fact_relation
        relation_paths_encountered_for_this_fact = set()
        for two_step_graph_path in two_step_graph_paths:
            (_, step_one_rel, _), (_, step_two_rel, _) = two_step_graph_path
            relation_paths_encountered_for_this_fact.add(step_one_rel + SEPARATOR + step_two_rel)
        for two_step_relation_path in relation_paths_encountered_for_this_fact:
            relation_2_two_step_relation_paths[fact_relation][two_step_relation_path] += 1

    if max_steps == 2:
        return relation_2_one_step_relation_paths, relation_2_two_step_relation_paths

    print("Computing frequencies of 3-step relation paths for dataset %s..." % dataset.name)
    # for each training fact (that MAY be a self-loop)
    for fact_key in train_fact_to_three_step_graph_paths:

        # get its relation
        _, fact_relation, _ = fact_key.split(SEPARATOR)

        # get all the 3-step GRAPH PATHS from the fact head to the fact tail
        three_step_graph_paths = train_fact_to_three_step_graph_paths[fact_key]

        # extract the set of 3-step RELATION PATHS from the 3-step GRAPH PATHS,
        # and for each one of them add 1 to the result dict under the fact_relation
        relation_paths_encountered_for_this_fact = set()
        for three_step_graph_path in three_step_graph_paths:
            (_, step_one_rel, _), (_, step_two_rel, _), (_, step_three_rel, _) = three_step_graph_path
            relation_paths_encountered_for_this_fact.add(SEPARATOR.join([step_one_rel, step_two_rel, step_three_rel]))
        for three_step_relation_path in relation_paths_encountered_for_this_fact:
            relation_2_three_step_relation_paths[fact_relation][three_step_relation_path] += 1

    return relation_2_one_step_relation_paths, \
           relation_2_two_step_relation_paths, \
           relation_2_three_step_relation_paths


def save(dataset, max_steps=3):

    """
    Compute for each relation in a dataset the relation paths paths of length 1 or 2
    that co-occur at least once with that relation, together with the number of times they co-occur,
    and save the found associations in a text file.

    The graph paths for training facts and test facts will be written into separate files.

    The file structure will be, in both cases:
        relation  ; [ r1:80 | r2:950 | ... ]
        relation  ; [ r1;r2:6789 | r8:INVERSE_r1:9876 | ... ]

    (whitespaces added for greater clarity)

    :param dataset: the dataset to compute the mappings for
    :param max_steps: the maximum number of steps per path to take into account.
                      The highest value allowed for max_steps is currently 3
    """

    if max_steps <= 0:
        raise Exception("Invalid value for \"max_steps\": " + str(max_steps))
    if max_steps > 3:
        raise Exception("The maximum value supported for \"max_steps\" is 3")


    if max_steps == 1:
        relation_2_one_step_relation_paths = compute(dataset, max_steps)
    elif max_steps == 2:
        relation_2_one_step_relation_paths, relation_2_two_step_relation_paths = compute(dataset, max_steps)
    else:
        relation_2_one_step_relation_paths, relation_2_two_step_relation_paths, relation_2_three_step_relation_paths = compute(dataset, max_steps)

    one_step_relation_path_lines = []
    two_step_relation_path_lines = []
    three_step_relation_path_lines = []

    one_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, ONE_STEP_RELATION_PATHS_FILENAME)
    two_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, TWO_STEP_RELATION_PATHS_FILENAME)
    three_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, THREE_STEP_RELATION_PATHS_FILENAME)

    # for each relation in the dataset:
    for relation in sorted(dataset.relationships):

        # unescape the relation (in Yago3 there are escaped parts that use ";", and we don't want that
        relation = html.unescape(relation)

        # get frequencies of all one step relation paths for that relation
        one_step_paths_and_counts = relation_2_one_step_relation_paths[relation].items()
        if len(one_step_paths_and_counts) == 0:
            one_step_relation_path_lines.append(relation + "\n")
        else:
            one_step_paths_and_counts = sorted(one_step_paths_and_counts, key=operator.itemgetter(1, 0), reverse=True)
            one_step_paths_and_counts_strings = [path_count_pair[0] + ":" + str(path_count_pair[1]) for path_count_pair in one_step_paths_and_counts]
            one_step_relation_path_lines.append(relation + SEPARATOR + RELATION_PATHS_IN_LIST_SEPARATOR.join(one_step_paths_and_counts_strings) + "\n")

        if max_steps >= 2:
            # get frequencies of all two step relation paths for that relation
            two_step_paths_and_counts = relation_2_two_step_relation_paths[relation].items()
            if len(two_step_paths_and_counts) == 0:
                two_step_relation_path_lines.append(relation + "\n")
            else:
                two_step_paths_and_counts = sorted(two_step_paths_and_counts, key=operator.itemgetter(1, 0), reverse=True)
                two_step_paths_and_counts_strings = [path_count_pair[0] + ":" + str(path_count_pair[1]) for path_count_pair in two_step_paths_and_counts]
                two_step_relation_path_lines.append(relation + SEPARATOR + RELATION_PATHS_IN_LIST_SEPARATOR.join(two_step_paths_and_counts_strings) + "\n")

        if max_steps == 3:
            # get frequencies of all two step relation paths for that relation
            three_step_paths_and_counts = relation_2_three_step_relation_paths[relation].items()
            if len(three_step_paths_and_counts) == 0:
                three_step_relation_path_lines.append(relation + "\n")
            else:
                three_step_paths_and_counts = sorted(three_step_paths_and_counts, key=operator.itemgetter(1, 0), reverse=True)
                three_step_paths_and_counts_strings = [path_count_pair[0] + ":" + str(path_count_pair[1]) for path_count_pair in three_step_paths_and_counts]
                three_step_relation_path_lines.append(relation + SEPARATOR + RELATION_PATHS_IN_LIST_SEPARATOR.join(three_step_paths_and_counts_strings) + "\n")

    print("Saving 1-step relation paths for dataset %s into location %s" % (dataset.name, one_step_relation_paths_filepath))
    with open(one_step_relation_paths_filepath, "w") as output_file:
        output_file.writelines(one_step_relation_path_lines)

    if max_steps >= 2:
        print("Saving 2-step relation paths for dataset %s into location %s" % (dataset.name, two_step_relation_paths_filepath))
        with open(two_step_relation_paths_filepath, "w") as output_file:
            output_file.writelines(two_step_relation_path_lines)

    if max_steps == 3:
        print("Saving 3-step relation paths for dataset %s into location %s" % (dataset.name, three_step_relation_paths_filepath))
        with open(three_step_relation_paths_filepath, "w") as output_file:
            output_file.writelines(three_step_relation_path_lines)


def read_one_step_relation_paths(dataset):
    one_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, ONE_STEP_RELATION_PATHS_FILENAME)
    print("Reading 1-step relation paths for dataset %s from location %s" % (dataset.name, one_step_relation_paths_filepath))
    relation_2_one_step_path_counts = dict()
    with open(one_step_relation_paths_filepath, "r") as one_step_rel_paths_input_file:
        # ex of line:      rel   p_rel_1:90,p_rel_2:65,p_rel_3:40
        lines = one_step_rel_paths_input_file.readlines()
        for line in lines:
            line = html.unescape(line).strip()
            path_2_count = defaultdict(lambda: 0)

            if not SEPARATOR in line:
                relation = line
            else:
                relation, path_2_count_str = line.strip().split(SEPARATOR, 1)
                for path_and_count in path_2_count_str.split(RELATION_PATHS_IN_LIST_SEPARATOR):
                    path, count = path_and_count.split(":")
                    count = int(count)
                    path_2_count[path] = count
            relation_2_one_step_path_counts[relation] = path_2_count

    return relation_2_one_step_path_counts


def read_two_step_relation_paths(dataset):
    two_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, TWO_STEP_RELATION_PATHS_FILENAME)

    print("Reading 2-step relation paths for dataset %s from location %s" % (dataset.name, two_step_relation_paths_filepath))
    relation_2_two_step_path_counts = dict()
    with open(two_step_relation_paths_filepath, "r") as two_step_rel_paths_input_file:
        # ex of line:      rel   p_rel_1;p_rel_2:90, p_rel_3;p_rel_4:65
        lines = two_step_rel_paths_input_file.readlines()
        for line in lines:
            line = html.unescape(line).strip()
            path_2_count = defaultdict(lambda: 0)

            if not SEPARATOR in line:
                relation = line
            else:
                relation, path_2_count_str = line.strip().split(SEPARATOR, 1)
                for path_and_count in path_2_count_str.split(RELATION_PATHS_IN_LIST_SEPARATOR):
                    path, count = path_and_count.split(":")
                    count = int(count)
                    path_2_count[path] = count
            relation_2_two_step_path_counts[relation] = path_2_count

    return relation_2_two_step_path_counts


def read_three_step_relation_paths(dataset):
    three_step_relation_paths_filepath = os.path.join(dataset.home, FOLDER, THREE_STEP_RELATION_PATHS_FILENAME)

    print("Reading 3-step relation paths for dataset %s from location %s" % (dataset.name, three_step_relation_paths_filepath))
    relation_2_three_step_path_counts = dict()
    with open(three_step_relation_paths_filepath, "r") as three_step_rel_paths_input_file:
        # ex of line:      rel   p_rel_1;p_rel_2;p_rel_3:90, p_rel_4;p_rel_5;p_rel_6:65
        lines = three_step_rel_paths_input_file.readlines()
        for line in lines:
            line = html.unescape(line).strip()
            path_2_count = defaultdict(lambda: 0)

            if not SEPARATOR in line:
                relation = line
            else:
                relation, path_2_count_str = line.strip().split(SEPARATOR, 1)
                for path_and_count in path_2_count_str.split(RELATION_PATHS_IN_LIST_SEPARATOR):
                    path, count = path_and_count.split(":")
                    count = int(count)
                    path_2_count[path] = count
            relation_2_three_step_path_counts[relation] = path_2_count

    return relation_2_three_step_path_counts

def read(dataset, max_steps=3):
    relation_2_one_step_path_counts = read_one_step_relation_paths(dataset)

    if max_steps == 1:
        return relation_2_one_step_path_counts

    relation_2_two_step_path_counts = read_two_step_relation_paths(dataset)
    if max_steps == 2:
        return relation_2_one_step_path_counts, relation_2_two_step_path_counts

    relation_2_three_step_path_counts = read_three_step_relation_paths(dataset)
    return relation_2_one_step_path_counts, relation_2_two_step_path_counts, relation_2_three_step_path_counts

#save(datasets.Dataset(datasets.FB15K))
#save(datasets.Dataset(datasets.FB15K_237))
#save(datasets.Dataset(datasets.WN18))
#save(datasets.Dataset(datasets.WN18RR), 3)
#save(datasets.Dataset(datasets.YAGO3_10))
