import os
import random
import datasets

HOME = "/Users/andrea/comparative_analysis/datasets/FB15k-randomized"

if not os.path.isdir(HOME):
    os.makedirs(HOME)

def _shuffle_relations(triples):

    relations = []
    for triple in triples:
        relations.append(triple[1])

    random.shuffle(relations)

    result = []
    for index in range(len(relations)):
        result.append((triples[index][0], relations[index], triples[index][2]))

    # verify the result is consistent to the input in terms of heads and tails
    for i in range(len(triples)):
        assert(triples[i][0] == result[i][0])
        assert(triples[i][2] == result[i][2])

    return result


dataset = datasets.Dataset(datasets.FB15K)

train_triples = dataset.train_triples
test_triples = dataset.test_triples
valid_triples = dataset.valid_triples

shuffled_train_triples = _shuffle_relations(train_triples)
shuffled_test_triples = _shuffle_relations(test_triples)
shuffled_valid_triples = _shuffle_relations(valid_triples)

train_filepath = os.path.join(HOME, "train.txt")
valid_filepath = os.path.join(HOME, "valid.txt")
test_filepath = os.path.join(HOME, "test.txt")

with open(train_filepath, "w") as outfile:
    outfile.writelines(["\t".join(triple) + "\n" for triple in shuffled_train_triples])

with open(valid_filepath, "w") as outfile:
    outfile.writelines(["\t".join(triple) + "\n" for triple in shuffled_valid_triples])

with open(test_filepath, "w") as outfile:
    outfile.writelines(["\t".join(triple) + "\n" for triple in shuffled_test_triples])
