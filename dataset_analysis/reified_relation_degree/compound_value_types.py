import os
import datasets

def intersection(list1, list2):
    return list(set(list1).intersection(list2))

from collections import defaultdict

FOLDER = "reified_relation_degree"
ALL_FACTS_WITH_CVTS_FILENAME = "all_facts_with_cvts.csv"

def read(dataset, return_cvt_2_facts=False):

    filepath = os.path.join(datasets.home_folder_for(dataset.name), FOLDER, ALL_FACTS_WITH_CVTS_FILENAME)
    with open(filepath, "r") as infile:
        lines = infile.readlines()

    if return_cvt_2_facts:
        cvt_2_facts = defaultdict(lambda: [])
        for line in lines:
            head, rel, tail, cvts = line.strip().split(";", 3)
            cvts = cvts[1:-1].split(";")
            for cvt in cvts:
                cvt_2_facts[cvt].append((head, rel, tail))
        return cvt_2_facts
    else:
        fact_2_cvts = dict()
        for line in lines:
            head, rel, tail, cvts = line.strip().split(";", 3)
            cvts = cvts[1:-1].split(";")
            fact_2_cvts[";".join([head, rel, tail])]=cvts
        return fact_2_cvts

def compute(dataset):

    # read all the lines from freebase-clean, and extract the corresponding facts
    print("Reading freebase-clean... ")
    with open("/Users/andrea/comparative_analysis/datasets/freebase/freebase-clean.txt", "r") as freebase_input:
        clean_facts = [line.strip().split("\t") for line in freebase_input.readlines()]

    print("Computing CVTs for " + dataset.name + "...")

    # build
    #   - a map that associates to each head;rel in freebase-clean the list of valid tails
    #   - a map that associates to each rel;tail in freebase-clean the list of valid heads
    print("\tBuilding maps from freebase-clean...")
    clean_head_and_rel_2_tails, clean_rel_and_tail_2_heads = defaultdict(lambda: []), defaultdict(lambda: [])
    for clean_head, clean_rel, clean_tail in clean_facts:
        clean_head_and_rel_2_tails[clean_head + ";" + clean_rel].append(clean_tail)
        clean_rel_and_tail_2_heads[clean_rel + ";" + clean_tail].append(clean_head)

    all_facts_in_current_dataset = dataset.train_triples + dataset.test_triples + dataset.valid_triples

    # for each reified fact < head rel1.rel2-tail >
    #   - split the reified fact into its "two halves": head-rel1 and rel2-tail
    #   - retrieve all the tails that in freebase-clean complete the "head-rel1" part
    #   - retrieve all the heads that in freebase-clean complete the "rel2-tail" part
    #   - their intersection is the list of compound value types for the original < head rel1.rel2-tail > reified fact
    reified_fact_2_cvts = dict()

    print("\tMapping each fact in " + dataset.name + " to the freebase-clean CVTs...")
    for i in range(len(all_facts_in_current_dataset)):

        if i%1000 == 0:
            print("\t\t%i..." % i)

        head, rel, tail = all_facts_in_current_dataset[i]
        if not "." in rel:
            continue

        rel1, rel2 = rel.split(".")
        cvts = intersection(clean_head_and_rel_2_tails[head + ";" + rel1],
                            clean_rel_and_tail_2_heads[rel2 + ";" + tail])
        reified_fact_2_cvts[";".join([head, rel, tail])] = cvts

    return reified_fact_2_cvts

def save(dataset):
    reified_fact_2_cvts = compute(dataset)

    print("Saving the freebase-clean CVTs for each fact in " + dataset.name + "...")
    output_lines = []
    for reified_fact in reified_fact_2_cvts:
        output_lines.append(";".join([reified_fact]) + ";[" + ";".join(reified_fact_2_cvts[reified_fact]) + "]\n")

    filepath = os.path.join(datasets.home_folder_for(dataset.name), FOLDER, ALL_FACTS_WITH_CVTS_FILENAME)

    with open(filepath, "w") as outfile:
        outfile.writelines(output_lines)

#save(datasets.Dataset(datasets.FB15K_237))