"""
for each fact in test set, get the count the head degree and tail degree seen in training set
Then, use these degree to to define the fine grained class of each fact.
"""
import html
import os
from collections import defaultdict

import datasets
from dataset_analysis.degrees import entity_degrees
from datasets import Dataset

FILENAME = "test_facts_with_degree_classes.csv"

INTERVALS = [(0, 1), (1, 4), (4, 25), (25, 100), (100, "inf")]
CLASSES = []
CLASS_2_INTERVALS = dict()

for head_degree_interval in INTERVALS:
    for tail_degree_interval in INTERVALS:
        class_name = str(head_degree_interval[0]) + "-" + str(head_degree_interval[1]) + "__" + str(tail_degree_interval[0]) + "-" + str(tail_degree_interval[1])
        CLASSES.append(class_name)
        CLASS_2_INTERVALS[class_name] = (head_degree_interval, tail_degree_interval)


def _fact_belongs_to_degree_class(head_degree, tail_degree, class_name):

    if class_name not in CLASSES:
        return False

    (head_degree_interval, tail_degree_interval) = CLASS_2_INTERVALS[class_name]

    head_degree_lower_bound = -1 if head_degree_interval[0] == 0 else head_degree_interval[0]
    head_degree_upper_bound = 10**9 if head_degree_interval[1] == 'inf' else head_degree_interval[1]

    tail_degree_lower_bound = -1 if tail_degree_interval[0] == 0 else tail_degree_interval[0]
    tail_degree_upper_bound = 10**9 if tail_degree_interval[1] == 'inf' else tail_degree_interval[1]

    return (head_degree_lower_bound < head_degree <= head_degree_upper_bound and
            tail_degree_lower_bound < tail_degree <= tail_degree_upper_bound)



def compute(dataset_name):
    """
    Compute the mappings  < degree class -> facts that belong to that degree class >
    from all the test facts of a dataset, given the dataset name.

    :param dataset_name: the name of the dataset to compute the mappings for
    :return: a dict that associates each degree class to the corresponding test facts
    """

    print("Computing the mappings <degree class -> list of test facts belonging to that degree class> for dataset %s ..." % dataset_name)

    degree_class_2_facts = defaultdict(lambda:[])

    # get the mappings entity mid -> degree
    _, _, mid_2_degree = entity_degrees.read(dataset_name)

    # for each test fact, get the head degree and tail degree
    dataset = Dataset(dataset_name)
    for (head, relation, tail) in dataset.test_triples:
        head_degree = mid_2_degree[head]
        tail_degree = mid_2_degree[tail]

        for degree_class in CLASSES:
            if _fact_belongs_to_degree_class(head_degree, tail_degree, degree_class):
                degree_class_2_facts[degree_class].append((head, relation, tail))
                break

    return degree_class_2_facts


def read(dataset_name, read_separator=";", return_fact_2_class=False):
    """
    Read the mappings <test fact -> degree class that it belongs to >
    from the corresponding file of the dataset with the given name
    and return them
        - either in the format < degree class -> test facts that belong to that degree class >
        - or in the format <test fact -> degree class that it belongs to >


    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :param return_fact_2_class: if true, return mappings in the format <test fact -> degree class that it belongs to >
                                otherwise, return mappings in the format in the format < degree class -> test facts that belong to that degree class >
    :return: the computed mappings
    """

    print("Reading the mappings <degree class -> list of test facts belonging to that degree class> for dataset %s ..." % dataset_name)
    datase_folder = datasets.home_folder_for(dataset_name)

    with open(os.path.join(datase_folder, FILENAME), "r") as input_file:

        if not return_fact_2_class:
            degree_class_2_facts = dict()
            for degree_class in CLASSES:
                degree_class_2_facts[degree_class] = []

            for line in input_file.readlines():
                line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
                head, relation, tail, degree_class = line.strip().split(read_separator)
                degree_class_2_facts[degree_class].append([head, relation, tail])
            return degree_class_2_facts

        else:
            fact_2_class = dict()
            for line in input_file.readlines():
                line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
                head, relation, tail, degree_class = line.strip().split(read_separator)
                fact_2_class[";".join([head, relation, tail])] = degree_class
            return fact_2_class


def save(dataset_name, write_separator=";"):
    """
    Compute the mappings < test fact -> degree class > for all the test facts of a dataset,
    and save them in a file.

    :param write_separator: the separator to use when writing the file
    :param dataset_name: the name of the dataset for which to compute the mappings
    """

    degree_class_2_facts = compute(dataset_name)
    lines = []
    for degree_class in CLASSES:
        for fact in degree_class_2_facts[degree_class]:
            head, relationship, tail = fact
            lines.append(write_separator.join([head, relationship, tail, degree_class]) + "\n")

    print("Saving the mappings <degree class -> list of test facts belonging to that degree class> for dataset %s ..." % dataset_name)
    dataset_home = datasets.home_folder_for(dataset_name)
    output_filepath = os.path.join(dataset_home, FILENAME)
    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)