import html
import os

FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"

ALL_DATASET_NAMES = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10]

ROOT = "/Users/andrea/comparative_analysis/"

class Dataset:

    def __init__(self, name, separator="\t"):

        self.name = name

        self.home = os.path.join(ROOT, "datasets", self.name)
        if not os.path.isdir(self.home):
            raise Exception("Folder %s does not exist" % self.home)

        self.train_path = os.path.join(self.home, "train.txt")
        self.valid_path = os.path.join(self.home, "valid.txt")
        self.test_path = os.path.join(self.home, "test.txt")

        self.entities = set()
        self.relationships = set()


        print("Reading train triples for %s..." % self.name)
        self.train_triples = self._read_triples(self.train_path, separator)
        print("Reading validation triples for %s..." % self.name)
        self.valid_triples = self._read_triples(self.valid_path, separator)
        print("Reading test triples for %s..." % self.name)
        self.test_triples = self._read_triples(self.test_path, separator)


    def _read_triples(self, triples_path, separator="\t"):
        triples = []
        with open(triples_path, "r") as triples_file:
            lines = triples_file.readlines()
            for line in lines:
                #line = html.unescape(line)
                head, relationship, tail = line.strip().split(separator)
                triples.append((head, relationship, tail))
                self.entities.add(head)
                self.entities.add(tail)
                self.relationships.add(relationship)

        return triples


def home_folder_for(dataset_name):
    dataset_home = os.path.join(ROOT, "datasets", dataset_name)
    if os.path.isdir(dataset_home):
        return dataset_home
    else:
        raise Exception("Folder %s does not exist" % dataset_home)
