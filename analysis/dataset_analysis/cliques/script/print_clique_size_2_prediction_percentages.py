from collections import defaultdict

from dataset_analysis.cliques import cliques
from dataset_analysis.siblings import sibling_classes
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10

max_clique_size_2_facts = cliques.read(YAGO3_10)

all = 0
for (max_clique_size, test_facts) in max_clique_size_2_facts.items():
    all += len(test_facts)

percs = []
for max_clique_size in sorted(max_clique_size_2_facts.keys()):
    predictions = len(max_clique_size_2_facts[max_clique_size])
    perc = 100*float(predictions)/float(all)
    perc = round(perc, 2)
    percs.append(perc)
    print(str(max_clique_size) + ": " + str(perc) + "%")
