import numpy as np
import matplotlib.pyplot as plt

import performances
from dataset_analysis.degrees import entity_degrees
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.paths import tfidf_support
from datasets import FB15K, YAGO3_10, Dataset

from io_utils import *
from models import TRANSE, ROTATE, SIMPLE, CONVE, ANYBURL

dataset_name = FB15K

test_fact_2_support = tfidf_support.read(Dataset(dataset_name))
buckets_to_count = defaultdict(lambda: 0)
for fact in test_fact_2_support:
    bucket = test_fact_2_support[fact] // 0.1
    buckets_to_count[bucket] += 1

for item in sorted(buckets_to_count.items(), key=lambda x:x[0]):
    print(str(item[0]/10) + "\t" + str(item[1]))

