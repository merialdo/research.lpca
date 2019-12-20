import html
import os
from collections import defaultdict

import datasets

FOLDER = "relation_properties"
FILENAME = "relation_properties.csv"

SYMMETRIC = "Symmetric"
ANTISYMMETRIC = "Antisymmetric"

REFLEXIVE = "Reflexive"
IRREFLEXIVE = "Irreflexive"

TRANSITIVE = "Transitive"

PREORDER = "Preorder"
ORDER = "Order"

PARTIAL_EQUIVALENCE = "Partial Equivalence"
EQUIVALENCE = "Equivalence"

ALL_PROPERTIES_NAMES = [REFLEXIVE, IRREFLEXIVE, SYMMETRIC, ANTISYMMETRIC, TRANSITIVE, PREORDER, ORDER, PARTIAL_EQUIVALENCE, EQUIVALENCE]



def _check_reflexivity(relation, relation_2_train_facts, tolerance=0.5):

    facts_with_that_relation = relation_2_train_facts[relation]

    heads = set()
    for fact in facts_with_that_relation:
        heads.add(fact[0])

    reflexive_count = 0
    overall_count = 0
    for head in heads:
        overall_count +=1
        if (head, relation, head) in facts_with_that_relation:
            reflexive_count+=1

    if reflexive_count == 0:
        return IRREFLEXIVE
    if (float(reflexive_count)/float(overall_count)) >= tolerance:
        return REFLEXIVE
    else:
        return None


def _check_symmetry(relation, relation_2_train_facts, tolerance=0.5):

    facts_with_that_relation = relation_2_train_facts[relation]

    numerator_count = 0
    denominator_count = 0
    for (head, relation, tail) in facts_with_that_relation:
        if head == tail:
            continue
        denominator_count += 1
        if (tail, relation, head) in facts_with_that_relation:
            numerator_count += 1

    if numerator_count == 0:
        return ANTISYMMETRIC
    if (float(numerator_count) / float(denominator_count)) >= tolerance:
        return SYMMETRIC
    else:
        return None


def _check_transitivity(relation, relation_2_train_facts, head_2_train_facts, tolerance=0.5):

    facts_with_that_relation = relation_2_train_facts[relation]

    all_chains_count = 0
    transitive_chains_count = 0

    for step_one_fact in facts_with_that_relation:
        (head1, relation1, tail1) = step_one_fact

        if head1 == tail1:
            continue

        for step_two_fact in head_2_train_facts[tail1]:
            (head2, relation2, tail2) = step_two_fact

            if relation2 != relation:
                continue
            if head2 == tail2 or tail2 == head1:
                continue

            all_chains_count += 1

            if (head1, relation, tail2) in facts_with_that_relation:
                transitive_chains_count += 1

    if transitive_chains_count > 0 and (float(transitive_chains_count) / float(all_chains_count)) >= tolerance:
        return TRANSITIVE
    else:
        return None



def compute(dataset):
    """
    Compute the mappings
            <relation -> types>
        from the training set of a specific dataset.

    :param dataset: the dataset to compute the mappings for
    :return: the computed mappings
    """

    relation_2_types = defaultdict(lambda: [])

    relation_2_train_facts = defaultdict(lambda: set())
    head_2_train_facts = defaultdict(lambda: set())

    print("Computing the mappings <relation name -> type> in %s training set..." % dataset.name)

    for (head, relation, tail) in dataset.train_triples:
        relation_2_train_facts[relation].add((head, relation, tail))
        head_2_train_facts[head].add((head, relation, tail))


    for relation in dataset.relationships:
        relation_2_types[relation] = []

        is_reflexive = _check_reflexivity(relation, relation_2_train_facts)
        is_symmetric = _check_symmetry(relation, relation_2_train_facts)
        is_transitive = _check_transitivity(relation, relation_2_train_facts, head_2_train_facts)

        if is_reflexive is not None:
            relation_2_types[relation].append(is_reflexive)

        if is_symmetric is not None:
            relation_2_types[relation].append(is_symmetric)
        if is_transitive is not None:
            relation_2_types[relation].append(is_transitive)

        if REFLEXIVE in relation_2_types[relation] and \
                TRANSITIVE in relation_2_types[relation]:
            relation_2_types[relation].append(PREORDER)
        if REFLEXIVE in relation_2_types[relation] and \
                ANTISYMMETRIC in relation_2_types[relation] and \
                TRANSITIVE in relation_2_types[relation]:
            relation_2_types[relation].append(ORDER)
        if SYMMETRIC in relation_2_types[relation] and \
                TRANSITIVE in relation_2_types[relation]:
            relation_2_types[relation].append(PARTIAL_EQUIVALENCE)
        if REFLEXIVE in relation_2_types[relation] and \
                SYMMETRIC in relation_2_types[relation] and \
                TRANSITIVE in relation_2_types[relation]:
            relation_2_types[relation].append(EQUIVALENCE)

    return relation_2_types


def read(dataset_name, read_separator=";"):
    """

    Read the file that contains the mappings <relationship name -> list of types> for the training set of a specific dataset.

    :param dataset_name: the name of the dataset for which to compute the mappings
    :param read_separator: the separator to use when reading the csv file
    :return: the computed mappings
    """

    print("Reading the mappings <relationship name -> list of types> in %s training set..." % dataset_name)
    dataset_home = datasets.home_folder_for(dataset_name)
    filepath = os.path.join(dataset_home, FOLDER, FILENAME)

    relation_2_types = dict()

    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            relation, types = line.strip().split(read_separator)
            relation_2_types[relation] = types.split(",")

    return relation_2_types


def save(dataset_name, write_separator = ";"):
    """
    Compute the mappings <relationship name -> list of types> for a specific dataset, and save them in a file.

    :param write_separator: the separator to use when writing the file
    :param dataset_name: the name of the dataset for which to compute the mappings
    """

    dataset = datasets.Dataset(dataset_name)
    relation_2_types = compute(dataset)

    lines = []
    for relation in sorted(dataset.relationships):
        lines.append(write_separator.join([relation, ",".join(relation_2_types[relation])]) + "\n")


    output_filepath = os.path.join(dataset.home, FOLDER, FILENAME)

    print("Writing the mappings <entity name -> in, out and overall degree> for %s training set in %s..." % (dataset_name, output_filepath))

    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)
