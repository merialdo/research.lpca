import html
import os
from collections import defaultdict

import datasets
from dataset_analysis.reified_relation_degree import compound_value_types

FOLDER = "reified_relation_degree"
TEST_FACTS_WITH_ARITY_FILENAME = "test_facts_with_reified_relation_degree.csv"

def read(dataset_name, read_separator=";", return_fact_2_arity=False):
    filepath = os.path.join(datasets.home_folder_for(dataset_name), FOLDER, TEST_FACTS_WITH_ARITY_FILENAME)
    with open(filepath, "r") as input_file:
        lines = input_file.readlines()

    if return_fact_2_arity:
        triple_2_arity = dict()

        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, arity = line.strip().split(read_separator)

            triple_2_arity[read_separator.join([head, relation, tail])] = int(arity)

        return triple_2_arity
    else:
        arity_2_triples = defaultdict(lambda:[])

        for line in lines:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, arity = line.strip().split(read_separator)

            arity_2_triples[int(arity)].append(read_separator.join([head, relation, tail]))

        return arity_2_triples

def _compute_ctv_degrees(dataset):
    ctv_2_facts = compound_value_types.read(dataset, return_cvt_2_facts=True)

    ctv_2_degree = dict()
    for ctv in ctv_2_facts:
        ctv_2_degree[ctv] = len(ctv_2_facts[ctv])

    return ctv_2_degree


def compute(dataset, read_separator=";"):
    print("Computing the arity for each test fact in " + dataset.name + "...")

    fact_2_ctvs = compound_value_types.read(dataset)
    ctv_2_degree = _compute_ctv_degrees(dataset)

    test_fact_2_arity = dict()

    for test_fact in dataset.test_triples:
        test_head, test_rel, test_tail = test_fact
        fact_key = ";".join(test_fact)

        # if the fact is reified, its arity is the degree of its ctv with the highest degree
        if "." in test_rel:  # "." in test_rel means that it is a reified fact
            ctvs = fact_2_ctvs[fact_key]

            if len(ctvs) > 0:
                test_fact_2_arity[fact_key] = max([ctv_2_degree[ctv] for ctv in ctvs])
            else:
                # this is weird but it can happen!
                # it means that no bridges were found in freebase for this fact, but it is still a reified fact.
                # in this case we arbitrarily set its arity to 2
                test_fact_2_arity[fact_key] = 2

        # we decide arbitrarily that facts that are not reified have arity 1
        else:
            test_fact_2_arity[fact_key] = 1

    return test_fact_2_arity

def save(dataset, read_separator=";"):
    test_fact_2_arity = compute(dataset)

    print("Saving the arity for each test fact in " + dataset.name + "...")
    output_lines = []
    for test_fact in dataset.test_triples:
        key = ";".join(test_fact)
        output_lines.append(key + ";" + str(test_fact_2_arity[key]) + "\n")

    filepath = os.path.join(datasets.home_folder_for(dataset.name), FOLDER, TEST_FACTS_WITH_ARITY_FILENAME)

    with open(filepath, "w") as outfile:
        outfile.writelines(output_lines)

#save(datasets.Dataset(datasets.FB15K_237))
