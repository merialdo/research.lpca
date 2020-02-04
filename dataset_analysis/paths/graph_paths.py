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



def compute(dataset, max_steps=3):
    """
        For both training facts and test facts of a specific dataset,
        extract all training 1-step graph paths, training 2-step and training 3-step graph paths
        connecting the head and tail of the input path

        Note: self-edges are considered acceptable in 1-step paths, but not in 2-step paths nor in 3-step paths.

        :param dataset: the dataset to compute the graph paths for
        :param max_steps: the maximum number of steps per path to take into account.
                           The highest value allowed for max_steps is currently 3
    """

    if max_steps <= 0:
        raise Exception("Invalid value for \"max_steps\": " + str(max_steps))
    if max_steps > 3:
        raise Exception("The maximum value supported for \"max_steps\" is 3")

    print("Computing graph paths for train and test facts in dataset %s" % dataset.name)

    train_fact_to_one_step_paths = dict()
    train_fact_to_two_step_paths = dict()
    train_fact_to_three_step_paths = dict()

    test_fact_to_one_step_paths = dict()
    test_fact_to_two_step_paths = dict()
    test_fact_to_three_step_paths = dict()

    print("\tPre-analysis for dataset %s" % dataset.name)
    entity_2_train_facts, entity_pair_2_train_facts = _pre_analysis(dataset)

    # get training 1-step graph paths and training 1-step graph paths for all training facts in the dataset
    print("\tComputing graph paths for train facts in dataset %s" % dataset.name)
    start = time.time()
    for i in range(len(dataset.train_triples)):
        fact = dataset.train_triples[i]

        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        fact = (html.unescape(fact[0]), html.unescape(fact[1]), html.unescape(fact[2]))

        train_fact_to_one_step_paths[SEPARATOR.join(fact)] = _compute_1_step_paths_for(fact, entity_pair_2_train_facts)
        if max_steps >= 2:
            train_fact_to_two_step_paths[SEPARATOR.join(fact)] = _compute_2_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)
        if max_steps == 3:
            train_fact_to_three_step_paths[SEPARATOR.join(fact)] = _compute_3_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)

        if i % 1000 == 0:
            end = time.time()
            interval = (float(round(end * 1000)) - float(round(start * 1000)))/1000
            print("\t\t%i training facts processed so far; time = %f" % (i, interval))
            start = end

    # get 1-step, 2-step and 3-step graph paths for all test facts in the dataset
    print("\tComputing all graph paths for test facts in dataset %s" % dataset.name)
    start = time.time()
    for i in range(len(dataset.test_triples)):
        fact = dataset.test_triples[i]

        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        fact = (html.unescape(fact[0]), html.unescape(fact[1]), html.unescape(fact[2]))

        test_fact_to_one_step_paths[SEPARATOR.join(fact)] = _compute_1_step_paths_for(fact, entity_pair_2_train_facts)
        if max_steps >= 2:
            test_fact_to_two_step_paths[SEPARATOR.join(fact)] = _compute_2_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)
        if max_steps == 3:
            test_fact_to_three_step_paths[SEPARATOR.join(fact)] = _compute_3_step_paths_for(fact, entity_2_train_facts, entity_pair_2_train_facts)

        if i % 1000 == 0:
            end = time.time()
            interval = (float(round(end * 1000)) - float(round(start * 1000)))/1000
            print("\t\t%i test facts processed so far; time = %f" % (i, interval))
            start = end

    if max_steps == 1:
        return train_fact_to_one_step_paths, test_fact_to_one_step_paths

    elif max_steps == 2:
        return train_fact_to_one_step_paths, train_fact_to_two_step_paths, \
               test_fact_to_one_step_paths, test_fact_to_two_step_paths
    else:
        return train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths, \
               test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths

def save(dataset, max_steps=3):
    """
        Compute for both the training facts and the test facts of a datasetm
        the graph paths of length 1 or 2 that connect the head and tail entity of each fact
        and save the found associations in a text file.

        The graph paths for training facts and test facts will be written into separate files.

        The file structure will be, in both cases:
            H ; r ; T ; [ H;s;T | H;t;T | ... ]
            H ; r ; T ; [ H;s;X;X;t;T | H;s1;Z;Z;t1;T |  ... ]

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
        train_fact_to_one_step_paths, test_fact_to_one_step_paths = compute(dataset, max_steps=max_steps)
    elif max_steps == 2:
        train_fact_to_one_step_paths, train_fact_to_two_step_paths, \
        test_fact_to_one_step_paths, test_fact_to_two_step_paths = compute(dataset, max_steps=max_steps)
    elif max_steps == 3:
        train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths, \
        test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths = compute(dataset, max_steps=max_steps)

    one_step_train_lines = []
    two_step_train_lines = []
    three_step_train_lines = []

    for train_fact in dataset.train_triples:

        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        train_fact = (html.unescape(train_fact[0]), html.unescape(train_fact[1]), html.unescape(train_fact[2]))

        train_fact_key = SEPARATOR.join(train_fact)
        one_step_paths = []
        two_step_paths = []
        three_step_paths = []

        for path in train_fact_to_one_step_paths[train_fact_key]:
            one_step_paths.append(SEPARATOR.join(path))
        one_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(one_step_paths)
        one_step_train_lines.append(train_fact_key + SEPARATOR + "[" + one_step_paths_str + "]" + "\n")

        if max_steps >= 2:
            for path in train_fact_to_two_step_paths[train_fact_key]:
                two_step_paths.append(SEPARATOR.join(path[0] + path[1]))
            two_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(two_step_paths)
            two_step_train_lines.append(train_fact_key + SEPARATOR + "[" + two_step_paths_str + "]" + "\n")

        if max_steps == 3:
            for path in train_fact_to_three_step_paths[train_fact_key]:
                three_step_paths.append(SEPARATOR.join(path[0] + path[1] + path[2]))
            three_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(three_step_paths)
            three_step_train_lines.append(train_fact_key + SEPARATOR + "[" + three_step_paths_str + "]" + "\n")

    one_step_test_lines = []
    two_step_test_lines = []
    three_step_test_lines = []

    for test_fact in dataset.test_triples:
        # unescape the fact (in Yago3 there are escaped parts that use ";", and we don't want that
        test_fact = (html.unescape(test_fact[0]), html.unescape(test_fact[1]), html.unescape(test_fact[2]))

        test_fact_key = SEPARATOR.join(test_fact)
        one_step_paths = []
        two_step_paths = []
        three_step_paths = []

        for path in test_fact_to_one_step_paths[test_fact_key]:
            one_step_paths.append(SEPARATOR.join(path))
        one_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(one_step_paths)
        one_step_test_lines.append(test_fact_key + SEPARATOR + "[" + one_step_paths_str + "]" + "\n")

        if max_steps >= 2:
            for path in test_fact_to_two_step_paths[test_fact_key]:
                two_step_paths.append(SEPARATOR.join(path[0] + path[1]))
            two_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(two_step_paths)
            two_step_test_lines.append(test_fact_key + SEPARATOR + "[" + two_step_paths_str + "]" + "\n")

        if max_steps == 3:
            for path in test_fact_to_three_step_paths[test_fact_key]:
                three_step_paths.append(SEPARATOR.join(path[0] + path[1] + path[2]))
            three_step_paths_str = PATHS_IN_LIST_SEPARATOR.join(three_step_paths)
            three_step_test_lines.append(test_fact_key + SEPARATOR + "[" + three_step_paths_str + "]" + "\n")

    output_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Saving one-step graph paths for train facts of dataset %s into location %s" % (dataset.name, output_filepath))
    with open(output_filepath, "w") as output_file:
        output_file.writelines(one_step_train_lines)

    output_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    print("Saving one-step graph paths for test facts of dataset %s into location %s" % (dataset.name, output_filepath))
    with open(output_filepath, "w") as output_file:
        output_file.writelines(one_step_test_lines)

    if max_steps >= 2:

        output_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
        print("Saving two-step graph paths for train facts of dataset %s into location %s" % (
        dataset.name, output_filepath))
        with open(output_filepath, "w") as output_file:
            output_file.writelines(two_step_train_lines)

        output_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
        print("Saving two-step graph paths for test facts of dataset %s into location %s" % (
        dataset.name, output_filepath))
        with open(output_filepath, "w") as output_file:
            output_file.writelines(two_step_test_lines)

    if max_steps == 3:
        output_filepath = os.path.join(dataset.home, FOLDER, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
        print("Saving three-step graph paths for train facts of dataset %s into location %s" % (dataset.name, output_filepath))
        with gzip.open(output_filepath, "wt") as output_file:
            output_file.writelines(three_step_train_lines)

        output_filepath = os.path.join(dataset.home, FOLDER, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
        print("Saving three-step graph paths for test facts of dataset %s into location %s" % (dataset.name, output_filepath))
        with gzip.open(output_filepath, "wt") as output_file:
            output_file.writelines(three_step_test_lines)


def read_one_step_paths(dataset, filename):
    input_filepath = os.path.join(dataset.home, FOLDER, filename)
    print("Reading one-step graph paths for facts of dataset %s from location %s" % (dataset.name, input_filepath))

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

def read_two_step_paths(dataset, filename):
    input_filepath = os.path.join(dataset.home, FOLDER, filename)
    print("Reading two-step graph paths for facts of dataset %s from location %s" % (dataset.name, input_filepath))

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


def read_three_step_paths(dataset, filename):
    input_filepath = os.path.join(dataset.home, FOLDER, filename)
    print("Reading three-step graph paths for facts of dataset %s from location %s" % (dataset.name, input_filepath))

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


def read_all(dataset, max_steps=3):
    train_fact_to_one_step_paths = read_one_step_paths(dataset, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    test_fact_to_one_step_paths = read_one_step_paths(dataset, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 1:
        return train_fact_to_one_step_paths, test_fact_to_one_step_paths

    test_fact_to_two_step_paths = read_two_step_paths(dataset, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    train_fact_to_two_step_paths = read_two_step_paths(dataset, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 2:
        return train_fact_to_one_step_paths, train_fact_to_two_step_paths, test_fact_to_one_step_paths, test_fact_to_two_step_paths

    test_fact_to_three_step_paths = read_three_step_paths(dataset, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    train_fact_to_three_step_paths = read_three_step_paths(dataset, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    return train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths, \
           test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths

def read_train(dataset, max_steps=3):

    train_fact_to_one_step_paths = read_one_step_paths(dataset, TRAIN_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 1:
        return train_fact_to_one_step_paths

    train_fact_to_two_step_paths = read_two_step_paths(dataset, TRAIN_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 2:
        return train_fact_to_one_step_paths, train_fact_to_two_step_paths

    train_fact_to_three_step_paths = read_three_step_paths(dataset, TRAIN_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    return train_fact_to_one_step_paths, train_fact_to_two_step_paths, train_fact_to_three_step_paths


def read_test(dataset, max_steps=3):
    test_fact_to_one_step_paths = read_one_step_paths(dataset, TEST_FACTS_WITH_ONE_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 1:
        return test_fact_to_one_step_paths

    test_fact_to_two_step_paths = read_two_step_paths(dataset, TEST_FACTS_WITH_TWO_STEP_GRAPH_PATHS_FILENAME)
    if max_steps == 2:
        return test_fact_to_one_step_paths, test_fact_to_two_step_paths

    test_fact_to_three_step_paths = read_three_step_paths(dataset, TEST_FACTS_WITH_THREE_STEP_GRAPH_PATHS_FILENAME)
    return test_fact_to_one_step_paths, test_fact_to_two_step_paths, test_fact_to_three_step_paths

#save(datasets.Dataset(datasets.FB15K))
#save(datasets.Dataset(datasets.FB15K_237))
#save(datasets.Dataset(datasets.WN18), 3)
#save(datasets.Dataset(datasets.WN18RR), 3)
#save(datasets.Dataset(datasets.YAGO3_10))
