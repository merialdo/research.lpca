"""
    This is an utility module to compute graph paths of 1, 2 and 3 steps between the entities of a fact.

    A GRAPH PATH is a path in a Knowledge Graph, going from a specific entity A to a specific entity Z.
    The path length is the amount of edges met in the path.

    For instance, A ---r1---> B ---r2---> Z is a 2-step graph path from A to Z.

    NOTE: in other modules, we refer to RELATION PATHS.
        RELATION PATHS are very different from GRAPH PATH, because they are just a sequences of relationships.
        A GRAPH PATH corresponds to one and only one RELATION PATH.
        On the contrary, the same RELATION PATH can have multiple GRAPH PATHS in the path.
        For instance, <r1, r2> is a 2-step RELATION PATH.

        For instance, given the subgraph

                |---place of birth---> Honolulu ----located in----|
                |                                                 |
                |                                                 |
            Barack Obama -------------nationality--------------> USA


        This is a GRAPH PATH:
            Barack Obama ---place of birth---> Honolulu ----located in----> USA

        This is a RELATION PATH:
            place of birth, located in
"""
import gc
import gzip
import html
import os
import time
from collections import defaultdict

import datasets

FOLDER = "paths"
TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME = "train_facts_with_one_step_graph_paths.csv"
TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME = "train_facts_with_two_step_graph_paths.csv"
TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME = "train_facts_with_three_step_graph_paths.csv.gz"
TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME = "test_facts_with_one_step_graph_paths.csv"
TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME = "test_facts_with_two_step_graph_paths.csv"
TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME = "test_facts_with_three_step_graph_paths.csv.gz"

PATHS_IN_LIST_SEPARATOR = "|"
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

def _pre_analysis(dataset):
    """
        Given a dataset, this private method computes and returns:
            - for each entity in the dataset, all the training facts that contain it
            - for each couple of entities in the dataset, all the training facts that connect them


        :param dataset: the dataset object to perform pre_analysis on

        :return: a map that associates each entity to the facts that contain it,
                 and a map that associates each couple of entity to the facts that connect them
    """
    entity_2_train_facts = defaultdict(lambda: [])
    entity_pair_2_train_facts = defaultdict(lambda: [])
    for train_triple in dataset.train_triples:
        head, rel, tail = train_triple

        # unescape the read triple (in Yago3 there are escaped parts that use ";", and we don't want that
        head, rel, tail = html.unescape(head), html.unescape(rel), html.unescape(tail)
        train_triple = (head, rel, tail)

        entity_2_train_facts[head].append(train_triple)
        entity_2_train_facts[tail].append(train_triple)
        entity_pair_2_train_facts[_build_key_for(head, tail)].append(train_triple)

    return entity_2_train_facts, entity_pair_2_train_facts



def _compute_1_step_paths_for(fact, entity_pair_2_train_facts):
    """
        This private method computes and returns all 1-step paths connecting the head and tail of a passed input fact.

        In greater detail, given an input fact <h, r, t> (that can be either a training, validation or test fact):
            - this method gets all the training facts connecting h and t.
            - it reformats them in order to always keep the original direction of the input fact.
                In other words, given an input fact A---r--->B,
                both facts like A---s--->B and like B---s--->A will contribute to the result list.
                        - A---s--->B and will be put in the result list as it is
                        - B---s--->A will be reformatted into A---INVERSE_s--->B before putting it in the result list
            - it returns the list of all reformatted facts, that are 1 step paths connecting h and r

        Each path is modeled as a list of facts; with 1-step paths, therefore, each list will only contain ONE fact.


        NOTES:
            - the input fact can belong to any set (test, valid, train)
            - the 1-step path facts, on the contrary, are extracted from the training set only

            - the input fact CAN be a self-loop (<h, r, h>)
                - the 1-step path facts, in this case, will be self-loops too

            - the input fact itself is NOT considered a valid 1-step path and therefore it is not included in the result list.

        :param fact: the fact <h, r, t> for which to extract all 1-step paths connecting h and t
        :param entity_pair_2_train_facts: a map that associates to each pair of entity
                                            the list of all the training facts connecting its head and tail

        :return: the list of 1-step paths connecting the head and tail of the passed input path
    """
    head, relation, tail = fact

    paths = []

    # get all training facts connecting h and t of the input fact, in any direction
    facts_containing_head_and_tail = entity_pair_2_train_facts[_build_key_for(head, tail)]

    # for each training fact connecting h and t, reformat it in order to match the original input fact direction
    for cur_fact in facts_containing_head_and_tail:
        cur_head, cur_rel, cur_tail = cur_fact

        # do not take into account the input fact itself
        if fact == cur_fact:
            continue

        # it the direction of this training fact is identical to the one of the input fact, there is no need to reformat
        if cur_head == head and cur_tail == tail:
            paths.append(cur_fact)
        # otherwise, reformat the fact inverting its head and tail and prepending "INVERSE_" to the relation
        elif cur_tail == head and cur_head == tail:
            paths.append((cur_tail, "INVERSE_" + cur_rel, cur_head))

    return paths

def _compute_2_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts):
    """
        This private method computes and returns all 2-step paths connecting the head and tail of a passed input fact.

        In greater detail, given an input fact <h, r, t> (that can be either a training, validation or test fact):
            - this method gets all the training facts connecting h with other any entity X: this is the step_1_fact
            - then it gets all the training facts connecting X with t: step_2_fact
            - it reformats all the (step_1_fact, step_2_fact) pairs to keep the original direction of the input fact.
                In other words, given an input fact H---r--->Q, these kinds of paths can potentially be identified:
                    [ H---s--->Q   Q----t--->T ]       ( that is:   H---s---> Q ---t---> T )
                    [ H---s--->Q   T----t--->Q ]       ( that is:   H---s---> Q <---t--- T )
                    [ Q---s--->H   Q----t--->T ]       ( that is:   H<---s--- Q ---t---> T )
                    [ Q---s--->H   T----t--->Q ]       ( that is:   H<---s--- Q <---t--- T )

                Before adding them to the result list, they will be reformatted in the following way:
                    [ H---s--->Q   Q----t--->T ]
                    [ H---s--->Q   Q----INVERSE_t--->T ]
                    [ H---INVERSE_s--->Q   Q----t--->T ]
                    [ H---INVERSE_s--->Q   Q----INVERSE_t--->T ]

            - it returns the list of all reformatted 2 step paths connecting h and r


        NOTES:
            - the input fact can belong to any set (test, valid, train)
            - the 2-step path facts, on the contrary, are extracted from the training set only

            - the input fact CAN be a self-loop (<h, r, h>)
            - the facts in the extracted 2-step paths CAN NOT be self loops
                (it would not be a real 2-step path)

        Each path is modeled as a list of facts; with 2-step paths, therefore, each list will contain TWO fact.
        The input fact itself, as well as self-loops, are NOT considered valid steps
        and therefore they are not used to build paths.

        :param fact: the fact <head, rel, ttail> for which to extract all 1 step paths connecting head and tail
        :param entity_2_train_facts: a map associating to each entity all the facts that involve it (in any direction)
        :param entity_pair_2_train_facts: a map associating to each pair of entities all the facts that connect them (in any direction)

        :return: the list of 2 step paths connecting the head and tail in any direction
    """

    paths = []

    input_fact_head, r, input_fact_tail = fact

    # get all the training facts like   <input_fact_head, *, X>   or   <X, *, input_fact_head>
    # (that is, all training facts containing h as either head or tail, excluding this fact itself).
    # each of them is a potential "step 1 fact" for our paths
    facts_containing_head = entity_2_train_facts[input_fact_head]
    if fact in facts_containing_head:
        facts_containing_head.remove(fact)


    # for each potential "step 1 fact":
    for step_one_fact in facts_containing_head:
        step_one_fact_head, step_one_fact_rel, step_one_fact_tail = step_one_fact

        # ignore "step 1 facts" that are self-loops
        if step_one_fact_head == step_one_fact_tail:
            continue

        # get the name of the entity X (the entity that is not input_fact_head) in this potential "step 1 fact"
        # also, reformat this potential "step 1 fact" in order to preserve the input fact direction
        intermediate_entity = None
        reformatted_step_1_fact = None
        if step_one_fact_head == input_fact_head:     # step 1 fact is <input_fact_head, *, X>
            intermediate_entity = step_one_fact_tail
            reformatted_step_1_fact = step_one_fact
        elif step_one_fact_tail == input_fact_head:   # step 1 fact is <X, *, input_fact_head>
            intermediate_entity = step_one_fact_head
            reformatted_step_1_fact = (step_one_fact_tail, "INVERSE_" + step_one_fact_rel, step_one_fact_head)
        assert(intermediate_entity is not None and reformatted_step_1_fact is not None)

        # ignore "step 1 facts" that lead to the tail directly (this would just lead to self-loops in step 2 fact)
        if intermediate_entity == input_fact_tail:
            continue

        # get all the training facts like <X, *, input_fact_tail> or <input_fact_tail, *, X>
        # (that is, all training facts connecting the intermediate entity to the tail of the input fact)
        # each of them is a "step 2 fact" for our paths
        step_two_facts = entity_pair_2_train_facts[_build_key_for(intermediate_entity, input_fact_tail)]

        # for each step 2 fact, reformat it
        # and add the pair (step 1 fact, step 2 fact) to the list of extracted 2-step paths
        for step_two_fact in step_two_facts:
            step_two_fact_head, step_two_fact_rel, step_two_fact_tail = step_two_fact

            # reformat this "step 2 fact" in order to preserve the input fact direction
            reformatted_step_2_fact = None
            if step_two_fact_head == intermediate_entity and step_two_fact_tail == input_fact_tail:
                reformatted_step_2_fact = step_two_fact
            elif step_two_fact_head == input_fact_tail and step_two_fact_tail == intermediate_entity:
                reformatted_step_2_fact = (step_two_fact_tail, "INVERSE_" + step_two_fact_rel, step_two_fact_head)
            assert(reformatted_step_2_fact is not None)

            # append to the result
            paths.append((reformatted_step_1_fact, reformatted_step_2_fact))


        # if the original fact is a self-loop, remove from the extracted 2-step paths
        # all the paths that just walk the same edge forth and back
        # e.g.: given the self-loop H ---r---> H
        #           if you have a fact H ---s---> K
        #           you can extract a (meaningless) path H ---s---> K ---inverse_s---> H
        # We want to remove these.
        # It is easy to detect them: just check if, given a self loop as input fact,
        # the path has same relationship and one time it is INVERSE_
        if input_fact_head == input_fact_tail:
            clean_paths = []
            for path in paths:
                (step_one_fact, step_two_fact) = path
                if not step_one_fact[1].replace("INVERSE_" ,"") == step_two_fact[1] and \
                        not step_two_fact[1].replace("INVERSE_", "") == step_one_fact[1]:
                    clean_paths.append(path)
            paths = clean_paths
    return paths



def _compute_3_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts):
    """
        This private method computes and returns all 3-step paths connecting the head and tail of a passed input fact.

        In greater detail, given an input fact <h, r, t>
            - this method gets all the training facts connecting h with other any entity X: this is the step_1_fact
                    where X not in {h, t}
            - then it gets all the training facts connecting X with Y: step_2_fact
                    where Y not in {h, t, X}
            - after that it gets all the training facts connecting Y with t: step_3_fact

            - it reformats all the (step_1_fact, step_2_fact) pairs to keep the original direction of the input fact.

                In other words, given an input fact H---r--->Q, these kinds of paths can potentially be identified:
                    [ H---s--->Q   Q----t--->T ]       ( that is:   H---s---> Q ---t---> T )
                    [ H---s--->Q   T----t--->Q ]       ( that is:   H---s---> Q <---t--- T )
                    [ Q---s--->H   Q----t--->T ]       ( that is:   H<---s--- Q ---t---> T )
                    [ Q---s--->H   T----t--->Q ]       ( that is:   H<---s--- Q <---t--- T )
                Before adding them to the result list, they will be reformatted in the following way:
                    [ H---s--->Q   Q----t--->T ]
                    [ H---s--->Q   Q----INVERSE_t--->T ]
                    [ H---INVERSE_s--->Q   Q----t--->T ]
                    [ H---INVERSE_s--->Q   Q----INVERSE_t--->T ]

            - it returns the list of all reformatted 2 step paths connecting h and r


        NOTES:
            - the input fact can belong to any set (test, valid, train) or even be a non-existing fact at all
            - the 3-step path facts, on the contrary, are extracted from the training set only

            - the input fact CAN be a self-loop (<h, r, h>)
            - the facts in the extracted 2-step paths CAN NOT be self loops
                (it would not be a real 2-step path)

        Each path is modeled as a list of facts; with 3-step paths, therefore, each list will contain THREE fact.
        The input fact itself, as well as self-loops, are NOT considered valid steps
        and therefore they are not used to build paths.

        :param fact: the fact <head, rel, tail> for which to extract all 1 step paths connecting head and tail
        :param entity_2_train_facts: a map associating to each entity all the facts that involve it (in any direction)
        :param entity_pair_2_train_facts: a map associating to each pair of entities all the facts that connect them (in any direction)

        :return: the list of 2 step paths connecting the head and tail in any direction
    """

    paths = []

    h, r, t = fact

    # get all the training facts like   <h, *, X>   or   <X, *, h>
    # (that is, all training facts containing h as either head or tail, excluding this fact itself).
    # each of them is a potential "step 1 fact" for our paths
    facts_containing_head = entity_2_train_facts[h]
    if fact in facts_containing_head:
        facts_containing_head.remove(fact)


    # for each potential "step 1 fact":
    for step_one_fact in facts_containing_head:
        step_one_fact_head, step_one_fact_rel, step_one_fact_tail = step_one_fact

        # find out which is h and which is X (first intermediate entity)
        # among step_one_fact_head and step_one_fact_tail
        x = None
        reverse = False
        if step_one_fact_head == h:
            x = step_one_fact_tail
        elif step_one_fact_tail == h:
            x = step_one_fact_head
            reverse = True  # if the tail of this step is h, the fact will have to be reversed
        assert (x is not None)

        # ignore "step 1 facts" that are self-loops, or that that lead to t itself
        if x == h or x == t:
            continue

        # reformat this potential "step 1 fact" in order to preserve the input fact direction
        reformatted_step_1_fact = (h, "INVERSE_" + step_one_fact_rel, x) if reverse else step_one_fact

        assert(x is not None)


        # get all the training facts like <X, *, Y> or <Y, *, X>
        # (that is, all training facts connecting the intermediate entity to the tail of the input fact)
        # each of them is a "step 2 fact" for our paths
        step_two_facts = entity_2_train_facts[x]

        # for each step 2 fact, reformat it
        # and add the pair (step 1 fact, step 2 fact) to the list of extracted 2-step paths
        for step_two_fact in step_two_facts:
            step_two_fact_head, step_two_fact_rel, step_two_fact_tail = step_two_fact

            # find out which is X (first intermediate entity) and which is Y (second intermediate entity)
            # among step_two_fact_head and step_two_fact_tail
            y = None
            reverse = False
            if step_two_fact_head == x:
                y = step_two_fact_tail
            elif step_two_fact_tail == x:
                y = step_two_fact_head
                reverse = True  # if the tail of this step is x, the fact will have to be reversed
            assert (y is not None)

            # ignore "step 2 facts" that are self-loops,
            # or that lead directly to t
            # or that lead back to h
            if y == x or y == t or y == h:
                continue

            reformatted_step_2_fact = (x, "INVERSE_" + step_two_fact_rel, y) if reverse else step_two_fact


            # get all the training facts like <Y, *, t> or <t, *, X>
            # (that is, all training facts connecting the intermediate entity to the tail of the input fact)
            # each of them is a "step 3 fact" for our paths
            step_three_facts = entity_pair_2_train_facts[_build_key_for(y, t)]

            # for each step 3 fact, reformat it
            # and add the triple (step 1 fact, step 2 fact, step 3 fact) to the list of extracted 3-step paths
            for step_three_fact in step_three_facts:
                step_three_fact_head, step_three_fact_rel, step_three_fact_tail = step_three_fact

                reverse = None
                if step_three_fact_head == y and step_three_fact_tail == t:
                    reverse = False
                elif step_three_fact_head == t and step_three_fact_tail == y:
                    reverse = True
                assert (reverse is not None)

                # reformat this "step 3 fact" in order to preserve the input fact direction
                reformatted_step_3_fact = (y, "INVERSE_" + step_three_fact_rel, t) if reverse else step_three_fact

                # append to the result
                paths.append((reformatted_step_1_fact, reformatted_step_2_fact, reformatted_step_3_fact))

    return paths



def _flush_paths_into_files(cur_batch,
                            fact_to_1_step_paths, fact_to_2_step_paths, fact_to_3_step_paths,
                            filepath_1_step, filepath_2_step, filepath_3_step):

    # FLUSH EVERYTHING FOUND SO FAR
    lines_1_step = []
    lines_2_step = []
    lines_3_step = []

    for cur_fact in cur_batch:
        cur_fact_key = SEPARATOR.join(cur_fact)
        paths_1 = [SEPARATOR.join(x) for x in fact_to_1_step_paths[cur_fact_key]]
        paths_1_str = PATHS_IN_LIST_SEPARATOR.join(paths_1)

        paths_2 = [SEPARATOR.join(x[0] + x[1]) for x in fact_to_2_step_paths[cur_fact_key]]
        paths_2_str = PATHS_IN_LIST_SEPARATOR.join(paths_2)

        paths_3 = [SEPARATOR.join(x[0] + x[1] + x[2]) for x in fact_to_3_step_paths[cur_fact_key]]
        paths_3_str = PATHS_IN_LIST_SEPARATOR.join(paths_3)

        lines_1_step.append(cur_fact_key + SEPARATOR + "[" + paths_1_str + "]" + "\n")
        lines_2_step.append(cur_fact_key + SEPARATOR + "[" + paths_2_str + "]" + "\n")
        lines_3_step.append(cur_fact_key + SEPARATOR + "[" + paths_3_str + "]" + "\n")

    with open(filepath_1_step, "a") as out_1_step:
        out_1_step.writelines(lines_1_step)
    with open(filepath_2_step, "a") as out_2_step:
        out_2_step.writelines(lines_2_step)
    with gzip.open(filepath_3_step, "at") as out_3_step:
        out_3_step.writelines(lines_3_step)


def compute_and_progressively_save(dataset):
    """
        For both training facts and test facts of a specific dataset,
        extract all training 1-step graph paths, training 2-step and training 3-step graph paths
        connecting the head and tail of the input path

        Note: self-edges are considered acceptable in 1-step paths, but not in 2-step paths nor in 3-step paths.

        :param dataset: the dataset to compute the graph paths for
    """

    print("Computing graph paths for train and test facts in dataset %s" % dataset.name)

    print("\tPre-analysis for dataset %s" % dataset.name)
    entity_2_train_facts, entity_pair_2_train_facts = _pre_analysis(dataset)

    # get training 1-step graph paths and training 1-step graph paths for all training facts in the dataset
    print("\tComputing graph paths for train facts in dataset %s" % dataset.name)


    output_train_filepath_one_step_paths = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    output_train_filepath_two_step_paths = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    output_train_filepath_three_step_paths = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)

    if os.path.isfile(output_train_filepath_one_step_paths):
        os.remove(output_train_filepath_one_step_paths)
    if os.path.isfile(output_train_filepath_two_step_paths):
        os.remove(output_train_filepath_two_step_paths)
    if os.path.isfile(output_train_filepath_three_step_paths):
        os.remove(output_train_filepath_three_step_paths)

    train_fact_to_one_step_paths = dict()
    train_fact_to_two_step_paths = dict()
    train_fact_to_three_step_paths = dict()

    current_train_fact_batch = []

    start = time.time()
    for i in range(len(dataset.train_triples)):
        fact = dataset.train_triples[i]

        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        fact = (html.unescape(fact[0]), html.unescape(fact[1]), html.unescape(fact[2]))
        current_train_fact_batch.append(fact)

        train_fact_to_one_step_paths[SEPARATOR.join(fact)] = _compute_1_step_paths_for(fact, entity_pair_2_train_facts)
        train_fact_to_two_step_paths[SEPARATOR.join(fact)] = _compute_2_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)
        train_fact_to_three_step_paths[SEPARATOR.join(fact)] = _compute_3_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)

        if i > 0 and i % 1000 == 0:
            # FLUSH EVERYTHING FOUND SO FAR
            _flush_paths_into_files(current_train_fact_batch,
                                    train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths,
                                    output_train_filepath_one_step_paths, output_train_filepath_two_step_paths, output_train_filepath_three_step_paths)

            # CLEAN THE DATA STRUCTURES AND GARBAGE COLLECTION TO FREE MEMORY
            train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths = dict(), dict(), dict()
            current_train_fact_batch = []
            gc.collect()

            # PRINT TIME
            end = time.time()
            interval = (float(round(end * 1000)) - float(round(start * 1000)))/1000
            print("\t\t%i training facts processed so far; time = %f" % (i, interval))
            start = end

    if len(dataset.train_triples) % 1000 != 0:
        _flush_paths_into_files(current_train_fact_batch,
                                train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths,
                                output_train_filepath_one_step_paths, output_train_filepath_two_step_paths, output_train_filepath_three_step_paths)

    # get 1-step, 2-step and 3-step graph paths for all test facts in the dataset
    print("\tComputing graph paths for test facts in dataset %s" % dataset.name)

    output_test_filepath_one_step_paths = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    output_test_filepath_two_step_paths = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    output_test_filepath_three_step_paths = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)

    if os.path.isfile(output_test_filepath_one_step_paths):
        os.remove(output_test_filepath_one_step_paths)
    if os.path.isfile(output_test_filepath_two_step_paths):
        os.remove(output_test_filepath_two_step_paths)
    if os.path.isfile(output_test_filepath_three_step_paths):
        os.remove(output_test_filepath_three_step_paths)

    test_fact_to_one_step_paths = dict()
    test_fact_to_two_step_paths = dict()
    test_fact_to_three_step_paths = dict()

    current_test_fact_batch = []

    start = time.time()
    for i in range(len(dataset.test_triples)):
        fact = dataset.test_triples[i]
        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        fact = (html.unescape(fact[0]), html.unescape(fact[1]), html.unescape(fact[2]))

        current_test_fact_batch.append(fact)

        test_fact_to_one_step_paths[SEPARATOR.join(fact)] = _compute_1_step_paths_for(fact, entity_pair_2_train_facts)
        test_fact_to_two_step_paths[SEPARATOR.join(fact)] = _compute_2_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)
        test_fact_to_three_step_paths[SEPARATOR.join(fact)] = _compute_3_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)

        if i > 0 and i % 1000 == 0:
            # FLUSH EVERYTHING FOUND SO FAR
            _flush_paths_into_files(current_test_fact_batch,
                                    test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths,
                                    output_test_filepath_one_step_paths, output_test_filepath_two_step_paths, output_test_filepath_three_step_paths)

            # CLEAN THE DATA STRUCTURES AND GARBAGE COLLECTION TO FREE MEMORY
            test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths = dict(), dict(), dict()
            current_test_fact_batch = []
            gc.collect()

            # PRINT TIME
            end = time.time()
            interval = (float(round(end * 1000)) - float(round(start * 1000))) / 1000
            print("\t\t%i test facts processed so far; time = %f" % (i, interval))
            start = end

    if len(dataset.test_triples) % 1000 != 0:
        _flush_paths_into_files(current_test_fact_batch,
                                test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths,
                                output_test_filepath_one_step_paths, output_test_filepath_two_step_paths, output_test_filepath_three_step_paths)


def _read_one_step_paths_from_file(input_filepath):

    fact_to_one_step_paths = defaultdict(lambda: [])

    with open(input_filepath, "r") as input_file:
        for line in input_file:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, relation, tail, one_step_paths = line.strip().split(SEPARATOR, 3)
            key = SEPARATOR.join([head, relation, tail])
            one_step_paths = one_step_paths[1:-1]

            if len(one_step_paths.strip()) == 0:
                one_step_paths = []
                fact_to_one_step_paths[key] = one_step_paths
            else:
                one_step_paths = one_step_paths.split(PATHS_IN_LIST_SEPARATOR)
                for one_step_path in one_step_paths:
                    h, r, t = one_step_path.split(SEPARATOR)
                    fact_to_one_step_paths[key].append((h, r, t))

    return fact_to_one_step_paths

def _read_two_step_paths_from_file(input_filepath):
    fact_to_two_step_paths = defaultdict(lambda: [])

    with open(input_filepath, "r") as input_file:
        for line in input_file:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, rel, tail, two_step_paths = line.strip().split(SEPARATOR, 3)
            two_step_paths = two_step_paths[1:-1]

            if two_step_paths == "":
                two_step_paths = []
            else:
                two_step_paths = two_step_paths.split(PATHS_IN_LIST_SEPARATOR)

            key = SEPARATOR.join([head, rel, tail])
            fact_to_two_step_paths[key] = []
            for two_step_path in two_step_paths:
                h1, r1, t1, h2, r2, t2 = two_step_path.split(SEPARATOR)
                fact_to_two_step_paths[key].append(((h1, r1, t1), (h2, r2, t2)))
    return fact_to_two_step_paths


def _read_three_step_paths_from_file(input_filepath):
    fact_to_three_step_paths = defaultdict(lambda: [])

    if input_filepath.endswith(".gz"):
        input_file = gzip.open(input_filepath, "rt")
    else:
        input_file = open(input_filepath, "r")

    try:
        for line in input_file:
            line = html.unescape(line)  # this may be needed by YAGO, that has some &amp; stuff
            head, rel, tail, three_step_paths = line.strip().split(SEPARATOR, 3)
            three_step_paths = three_step_paths[1:-1]
            if three_step_paths == "":
                three_step_paths = []
            else:
                three_step_paths = three_step_paths.split(PATHS_IN_LIST_SEPARATOR)

            key = SEPARATOR.join([head, rel, tail])
            fact_to_three_step_paths[key] = []
            for three_step_path in three_step_paths:
                h1, r1, t1, h2, r2, t2, h3, r3, t3 = three_step_path.split(SEPARATOR)
                fact_to_three_step_paths[key].append(((h1, r1, t1), (h2, r2, t2), (h3, r3, t3)))
    finally:
        input_file.close()

    return fact_to_three_step_paths




def read_all(dataset):
    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading one-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_one_step_paths = _read_one_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    print("Reading two-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_two_step_paths = _read_two_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading three-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_three_step_paths = _read_three_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading one-step graph paths for test facts of dataset %s into location %s" % (dataset.name, input_filepath))
    test_fact_to_one_step_paths = _read_one_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    print("Reading two-step graph paths for test facts of dataset %s into location %s" % (dataset.name, input_filepath))
    test_fact_to_two_step_paths = _read_two_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading three-step graph paths for test facts of dataset %s into location %s" % (dataset.name, input_filepath))
    test_fact_to_three_step_paths = _read_three_step_paths_from_file(input_filepath)

    return train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths, \
           test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths

def read_train(dataset):
    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading one-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_one_step_paths = _read_one_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    print("Reading two-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_two_step_paths = _read_two_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading three-step graph paths for train facts of dataset %s into location %s" % (dataset.name, input_filepath))
    train_fact_to_three_step_paths = _read_three_step_paths_from_file(input_filepath)

    return train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths


def read_test(dataset):

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading one-step graph paths for test facts of dataset %s from location %s" % (dataset.name, input_filepath))
    test_fact_to_one_step_paths = _read_one_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    print("Reading two-step graph paths for test facts of dataset %s from location %s" % (dataset.name, input_filepath))
    test_fact_to_two_step_paths = _read_two_step_paths_from_file(input_filepath)

    input_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    print("Reading three-step graph paths for test facts of dataset %s into location %s" % (dataset.name, input_filepath))
    test_fact_to_three_step_paths = _read_three_step_paths_from_file(input_filepath)

    return test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths

compute_and_progressively_save(datasets.Dataset(datasets.FB15K))
#compute_and_progressively_save(datasets.Dataset(datasets.FB15K_237))
#compute_and_progressively_save(datasets.Dataset(datasets.WN18))
#compute_and_progressively_save(datasets.Dataset(datasets.WN18RR))
#compute_and_progressively_save(datasets.Dataset(datasets.YAGO3_10))
