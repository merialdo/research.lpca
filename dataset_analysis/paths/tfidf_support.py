"""
    This is an utility module to measure the support to any test fact,
    based on the occurrence of specific 1-step-paths and 2-step-facts between its head and tail.
"""
import html
import os
import time
from collections import defaultdict
import numpy as np

import datasets
from dataset_analysis.paths import relation_paths, graph_paths

FOLDER = "paths"
TEST_FACT_2_SUPPORT_MAX1_FILENAME = "test_facts_with_max1_RPS.csv"
TEST_FACT_2_SUPPORT_MAX2_FILENAME = "test_facts_with_max2_RPS.csv"
TEST_FACT_2_SUPPORT_MAX3_FILENAME = "test_facts_with_max3_RPS.csv"
SEPARATOR=";"

def _cosine_similarity(vector_a, vector_b):
    """
        This private method computes the cosine similarity between two vectors

        :param vector_a: the first vector to compute the cosine similarity of
        :param vector_b: the second vector to compute the cosine similarity of

        :return: the computed cosine similarity
    """
    if vector_a.any() > 0 and vector_b.any() > 0:
        return np.dot(vector_a, vector_b)/(np.linalg.norm(vector_a)*np.linalg.norm(vector_b))
    else:
        return 0.0

def compute(dataset, max_steps=3):

    print("Computing tf-idf support for each test fact in dataset %s..." % dataset.name)

    #initialize as empty dicts
    relation_2_one_step_relation_path_counts = defaultdict(lambda:[])
    relation_2_two_step_relation_path_counts = defaultdict(lambda:[])
    relation_2_three_step_relation_path_counts = defaultdict(lambda:[])
    test_fact_2_one_step_graph_paths = defaultdict(lambda:[])
    test_fact_2_two_step_graph_paths = defaultdict(lambda:[])
    test_fact_2_three_step_graph_paths = defaultdict(lambda:[])

    if max_steps == 1:
        # read, for all relations, the frequencies of all 1-step relation paths in the training set
        relation_2_one_step_relation_path_counts = relation_paths.read(dataset, max_steps)
        # read, for each test fact, the training 1-step, graph paths connecting its head with its tail
        test_fact_2_one_step_graph_paths = graph_paths.read_test(dataset, max_steps)
    elif max_steps == 2:
        # read, for all relations, the frequencies of all 1-step and 2-step relation paths in the training set
        relation_2_one_step_relation_path_counts, relation_2_two_step_relation_path_counts = relation_paths.read(dataset, max_steps)
        # read, for each test fact, the training 1-step and 2-step graph paths connecting its head with its tail
        test_fact_2_one_step_graph_paths, test_fact_2_two_step_graph_paths = graph_paths.read_test(dataset, max_steps)
    else:
        # read, for all relations, the frequencies of all 1-step, 2-step and 3-step relation paths in the training set
        relation_2_one_step_relation_path_counts, relation_2_two_step_relation_path_counts, relation_2_three_step_relation_path_counts = relation_paths.read(dataset)
        # read, for each test fact, the training 1-step, 2-step and 3-step graph paths connecting its head with its tail
        test_fact_2_one_step_graph_paths, test_fact_2_two_step_graph_paths, test_fact_2_three_step_graph_paths = graph_paths.read_test(dataset)

    # for each relation, get the list of all 1-step, 2-step, and 3-step relation paths
    # co-occurring at least once with that relation of all relation, sorted alphabetically
    all_paths = set()
    for relation in dataset.relationships:

        # unescape the relation (in Yago3 there are escaped parts that use ";", and we don't want that
        relation = html.unescape(relation)

        #for path in relation_2_one_step_relation_path_counts[relation]:
        #    all_paths.add(path)
        #for path in relation_2_two_step_relation_path_counts[relation]:     #empty if max_steps < 2
        #    all_paths.add(path)
        for path in relation_2_three_step_relation_path_counts[relation]:   #empty if max_steps < 3
            all_paths.add(path)

    all_paths = sorted(list(all_paths))


    # ======= 1) COMPUTE DF (document frequency) FOR EACH PATH UNDER ANALYSIS
    print("\tComputing df for each test relation path in dataset %s..." % dataset.name)
    # DF = frequency across all documents (all relations)

    # map each relation path to the number of relations that it co-occurs with
    path_2_df = defaultdict(lambda:0.0)
    for relation in dataset.relationships:

        # unescape the relation (in Yago3 there are escaped parts that use ";", and we don't want that
        relation = html.unescape(relation)

        #for one_step_path in relation_2_one_step_relation_path_counts[relation]:
        #    path_2_df[one_step_path] += 1.0
        #for two_step_path in relation_2_two_step_relation_path_counts[relation]:       #empty if max_steps < 2
        #    path_2_df[two_step_path] += 1.0
        for three_step_path in relation_2_three_step_relation_path_counts[relation]:        #empty if max_steps < 3
            path_2_df[three_step_path] += 1.0


    # ======= 2) COMPUTE TF FOR EACH COUPLE <RELATION, PATH>

    print("\tComputing TF for each relation and relation path in dataset %s..." % dataset.name)

    # for each relation path (= "word), for each relation type (= "document"),
    #   TF = freq of relation path under that relation type / sum of all relation path frequencies under that relation type

    # map each relation to a nested dict
    #   that maps each path to the tf for that relation and that path
    relation_2_path_tf = dict()

    # for each relation
    for relation in dataset.relationships:

        # unescape the relation (in Yago3 there are escaped parts that use ";", and we don't want that
        relation = html.unescape(relation)

        relation_2_path_tf[relation] = dict()

        # get the frequencies of 1-step, 2-step and 3-step relation paths with respect to the current relation
        #one_step_path_2_count_in_current_relation = relation_2_one_step_relation_path_counts[relation]
        #two_step_path_2_count_in_current_relation = relation_2_two_step_relation_path_counts[relation]  #empty if max_steps < 2
        three_step_path_2_count_in_current_relation = relation_2_three_step_relation_path_counts[relation]  #empty if max_steps < 3

        overall_path_count_in_cur_rel = 0
        # get the sum of frequencies of all paths
        #overall_path_count_in_cur_rel = float(sum(one_step_path_2_count_in_current_relation.values()))
        #if max_steps >= 2:
        #    overall_path_count_in_cur_rel += sum(two_step_path_2_count_in_current_relation.values())
        if max_steps == 3:
            overall_path_count_in_cur_rel += sum(three_step_path_2_count_in_current_relation.values())

        # for each 1-step path or 2-step path or 3-step path
        # compute the TF as  path frequency / sum of frequencies of all paths
        #for one_step_path in one_step_path_2_count_in_current_relation:
        #    count = float(one_step_path_2_count_in_current_relation[one_step_path])
        #    relation_2_path_tf[relation][one_step_path] = count/overall_path_count_in_cur_rel
        #for two_step_path in two_step_path_2_count_in_current_relation:         # empty if max_steps < 2
        #    count = float(two_step_path_2_count_in_current_relation[two_step_path])
        #    relation_2_path_tf[relation][two_step_path] = count/overall_path_count_in_cur_rel
        for three_step_path in three_step_path_2_count_in_current_relation:     # empty if max_steps < 3
            count = float(three_step_path_2_count_in_current_relation[three_step_path])
            relation_2_path_tf[relation][three_step_path] = count/overall_path_count_in_cur_rel


    # ======= 3) COMPUTE IDF FOR EACH PATH

    print("\tComputing IDF for each relation path in dataset %s..." % dataset.name)

    # IDF = for each relation path (= "word")
    #         log(    num of relation types (= "documents") / DF of that relation path across all relation types     )

    # map each relation path to its IDF
    # computed as log(number of relations / number of relations co-occurring with that path)
    path_2_idf = dict()
    all_relations = float(len(dataset.relationships))
    for path in path_2_df:
        path_2_idf[path] = np.log(all_relations/float(path_2_df[path]))


    # ======= 4) COMPUTE TF-IDF VECTOR FOR EACH RELATION

    print("\tComputing TF-IDF vector for each relation in dataset %s..." % dataset.name)

    # compute for each relation a vector containing the TF-IDF value for all of the relation paths
    relation_2_tfidf_vec = dict()
    for relation in relation_2_path_tf:
        # initialize the tf-idf vector with zeroes
        relation_2_tfidf_vec[relation] = np.zeros(len(all_paths), dtype=np.float)
        # for each relation path if the relation path co-occurs with the relation
        # update the position of the path in the relation vector with the corresponding tf-idf value for the relation
        path_2_tf = relation_2_path_tf[relation]
        for i in range(len(all_paths)):
            path = all_paths[i]
            if path in path_2_tf:
                value = path_2_tf[path] * path_2_idf[path]
                relation_2_tfidf_vec[relation][i] = value

    # ======= 4) COMPUTE TF-IDF VECTOR FOR EACH TEST FACT


    print("\tComputing TF-IDF vector for each test fact in dataset %s..." % dataset.name)

    test_fact_2_tfidf_vec = dict()

    # compute for each relation a vector containing the TF-IDF value
    # for all of the relation paths that connect its head with its tail
    # (
    start = time.time()
    for test_fact_index in range(len(dataset.test_triples)):
        end = time.time()

        if test_fact_index % 1000 == 0:
            interval = float(end * 1000 - start * 1000)/1000
            print("\t\ttest fact: %i; time = %f" % (test_fact_index, interval))
            start = time.time()

        (test_head, test_rel, test_tail) = dataset.test_triples[test_fact_index]

        # unescape the read fact (in Yago3 there are escaped parts that use ";", and we don't want that
        test_head, test_rel, test_tail = html.unescape(test_head), html.unescape(test_rel), html.unescape(test_tail)

        test_fact_key = SEPARATOR.join([test_head, test_rel, test_tail])

        # get all the relation paths that connect the head and the tail of this test fact
        relation_paths_for_this_test_fact = set()
        #for (one_step_path_head, one_step_path_rel, one_step_path_tail) in test_fact_2_one_step_graph_paths[test_fact_key]:
        #    relation_paths_for_this_test_fact.add(one_step_path_rel)
        #for ((two_step_path_head_1, two_step_path_rel_1, two_step_path_tail_1),
        #     (two_step_path_head_2, two_step_path_rel_2, two_step_path_tail_2)) in test_fact_2_two_step_graph_paths[test_fact_key]: # empty if max_steps < 2
        #    relation_paths_for_this_test_fact.add(two_step_path_rel_1 + SEPARATOR + two_step_path_rel_2)
        for ((three_step_path_head_1, three_step_path_rel_1, three_step_path_tail_1),
             (three_step_path_head_2, three_step_path_rel_2, three_step_path_tail_2),
             (three_step_path_head_3, three_step_path_rel_3, three_step_path_tail_3)) in test_fact_2_three_step_graph_paths[test_fact_key]: # empty if max_steps < 3
            relation_paths_for_this_test_fact.add(SEPARATOR.join([three_step_path_rel_1, three_step_path_rel_2, three_step_path_rel_3]))


        relation_paths_for_this_test_fact = list(relation_paths_for_this_test_fact)

        # initialize the TF-IDF vector for this test fact with zeros
        tfidf_vector = np.zeros(len(all_paths), dtype=np.float)
        for i in range(len(all_paths)):
            # for each path, if the path co-occurs with this test fact,
            #   - TF = freq of the path to this test fact / sum of frequencies of all relation paths to this fact
            #        NOTE:  this is just a TEST FACT, and not a relation path with multiple instances (graph paths).
            #               As a consequence, relation path frequencies to this fact are
            #                        1 if the path co-occurs with the test fact
            #                        0 otherwise
            #               Therefore:
            #                   freq of the path to this fact = 1
            #                   sum of frequencies of paths to this fact = number of paths co-occurring with this fact
            #   - IDF = log(num of documents / num of documents this relation path co-occurs at least once)
            #        NOTE:  num of documents = num of relations + 1,
            #               num of documents this path co-occurs at least once = path_2_df[path] + 1
            #                   (the +1 are added because we are also considering this additional fact)
            path = all_paths[i]
            if path in relation_paths_for_this_test_fact:
                tf = 1.0/float(len(relation_paths_for_this_test_fact))
                idf = np.log(float(all_relations+1.0)/float(path_2_df[path]+1.0))
                tfidf_vector[i] = tf*idf
            # (otherwise, the TF-IDF value of the path with this test fact is just zero, so don't update anything)

        test_fact_2_tfidf_vec[test_fact_key]=tfidf_vector

    # ======= 5) COMPUTE COSINE SIMILARITIES
    test_fact_2_support = dict()

    print("\tComputing cosine similarity between the vector of each test fact and the vector of the corresponding relation %s..." % dataset.name)

    for test_fact_key in test_fact_2_tfidf_vec:
        head, relation, tail = test_fact_key.split(SEPARATOR)
        test_fact_2_support[test_fact_key] = _cosine_similarity(relation_2_tfidf_vec[relation], test_fact_2_tfidf_vec[test_fact_key])

    return test_fact_2_support


def save(dataset, max_steps=3):

    if max_steps == 1:
        filename = TEST_FACT_2_SUPPORT_MAX1_FILENAME
    elif max_steps == 2:
        filename = TEST_FACT_2_SUPPORT_MAX2_FILENAME
    else:
        filename = TEST_FACT_2_SUPPORT_MAX3_FILENAME

    test_fact_2_support = compute(dataset, max_steps)

    lines = []

    for test_triple in dataset.test_triples:

        # unescape the read fact (in Yago3 there are escaped parts that use ";", and we don't want that
        test_triple = html.unescape(test_triple[0]), html.unescape(test_triple[1]), html.unescape(test_triple[2])

        key = SEPARATOR.join(test_triple)
        lines.append(key + SEPARATOR + str(test_fact_2_support[key]) + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, filename)
    print("Saving support for each test fact in dataset %s into location %s" % (dataset.name, output_filepath))
    with open(output_filepath, "w") as output_file:
        output_file.writelines(lines)


def read(dataset, max_steps=3):
    test_fact_2_support = dict()

    if max_steps==1:
        filename = TEST_FACT_2_SUPPORT_MAX1_FILENAME
    elif max_steps==2:
        filename = TEST_FACT_2_SUPPORT_MAX2_FILENAME
    else:
        filename = TEST_FACT_2_SUPPORT_MAX3_FILENAME
    input_filepath = os.path.join(dataset.home, FOLDER, filename)
    print("Reading relation paths support for each test fact in dataset %s from location %s" % (dataset.name, input_filepath))


    with open(input_filepath, "r") as input_file:
        for line in input_file:
            line = html.unescape(line)
            head, rel, tail, support_str = line.strip().split(SEPARATOR)
            support = float(support_str)

            test_fact_2_support[SEPARATOR.join([head, rel, tail])] = support

    return test_fact_2_support

#save(datasets.Dataset(datasets.FB15K), 1)
#save(datasets.Dataset(datasets.FB15K_237), 1)
#save(datasets.Dataset(datasets.WN18), 3)
#save(datasets.Dataset(datasets.WN18RR), 3)
#save(datasets.Dataset(datasets.YAGO3_10), 1)
