from dataset_analysis.degrees import relation_mentions
from dataset_analysis.relation_cardinalities import relation_cardinalities, relation_fine_classes
from datasets import Dataset, FB15K

dataset = Dataset(FB15K)
rel_2_cardinality_dicts = relation_cardinalities.compute(dataset)

fine_class_2_rels = relation_fine_classes.read(FB15K)
rel_2_mentions = relation_mentions.read(FB15K)


all = 0
for key in rel_2_mentions:
    all += rel_2_mentions[key]

for fine_class in fine_class_2_rels:
    mentions = 0
    rels = fine_class_2_rels[fine_class]
    for rel in rels:
        mentions += rel_2_mentions[rel]

    perc = 100*float(mentions)/float(all)
    perc = round(perc, 2)
    print(fine_class)
    print(str(perc) + "%")
    print()