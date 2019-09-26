import math
from collections import defaultdict

from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import degree_classes
from datasets import FB15K, Dataset

dataset = Dataset(FB15K)

test_fact_2_degree_class = degree_classes.read(FB15K, return_fact_2_class=True)
_, _, mid_2_degree = entity_degrees.read(FB15K)

degree_class_2_count = defaultdict(lambda: 0)
all_count = len(test_fact_2_degree_class)

for test_fact in test_fact_2_degree_class:
    degree_class = test_fact_2_degree_class[test_fact]
    degree_class_2_count[degree_class] += 1

for degree_class in degree_class_2_count:
    perc = 100*float(degree_class_2_count[degree_class])/float(all_count)
    perc = round(perc, 2)
    print(degree_class)
    print(str(perc) + "%")
    print()