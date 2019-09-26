import numpy as np
import matplotlib.pyplot as plt
from io_utils import *


def plot_dict(dictionary, title, xlabel, ylabel, xticks):
    x = []
    y = []

    items = sorted(dictionary.items(), key=lambda x: x[0])
    for i in range(len(items)):
        (count, amount_of_elements_with_that_count) = items[i]
        x.append(int(count))
        y.append(int(amount_of_elements_with_that_count))

    plt.scatter(x, y, s=2)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(xticks)
    plt.yscale('log')
    plt.xscale('log')

    plt.grid(True)
    plt.show()

def plot_dicts(dictionary1, dictionary2, title, xlabel, ylabel, xticks):
    x1 = []
    y1 = []
    items1 = sorted(dictionary1.items(), key=lambda x: x[1])
    for i in range(len(items1)):
        (count, amount_of_elements_with_that_count) = items1[i]
        x1.append(int(count))
        y1.append(int(amount_of_elements_with_that_count))
    plt.scatter(x1, y1, s=2)

    x2 = []
    y2 = []
    items2 = sorted(dictionary2.items(), key=lambda x: x[1])
    for i in range(len(items2)):
        (count, amount_of_elements_with_that_count) = items2[i]
        x2.append(int(count))
        y2.append(int(amount_of_elements_with_that_count))
    plt.scatter(x2, y2, s=2)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks([])
    plt.xscale('log')
    plt.yscale('log')


    plt.grid(True)


    plt.show()

def remove_entity_from_facts(facts, entity):
    new_facts = []
    for fact in facts:
        if fact["head"] != entity and fact["tail"] != entity:
            new_facts.append(fact)

    return new_facts


def get_richest_entity_from_facts(facts):

    entity_2_degree = defaultdict(lambda:0)

    for fact in facts:
        entity_2_degree[fact["head"]] += 1
        entity_2_degree[fact["tail"]] += 1

    item = (sorted(entity_2_degree.items(), key=lambda item:item[1], reverse=True))[0]
    print(item)
    return item[0], item[1]


facts = get_facts_from_openke_facts_files(["/Users/andrea/paper/WN18/train2id.txt", "/Users/andrea/paper/WN18/valid2id.txt", "/Users/andrea/paper/WN18/test2id.txt"])
amount_of_remaining_facts = len(facts)
threshold = (amount_of_remaining_facts/100)*20

erased_entities = []
while(amount_of_remaining_facts > threshold):
    richest, degree = get_richest_entity_from_facts(facts)
    facts = remove_entity_from_facts(facts, richest)
    erased_entities.append(richest)
    amount_of_remaining_facts = len(facts)
print(len(erased_entities))
