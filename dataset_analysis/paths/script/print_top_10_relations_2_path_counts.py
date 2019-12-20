import datasets
from dataset_analysis.degrees import relation_mentions
from dataset_analysis.paths import relation_paths

rel_2_mentions = relation_mentions.read(datasets.FB15K)
rel_2_paths_counts = relation_paths.read(datasets.Dataset(datasets.FB15K))



rel_2_mentions_items = sorted(rel_2_mentions.items(), key=lambda x:x[1], reverse=True)
for relation_and_mentions in rel_2_mentions_items[0:1]:
    relation = relation_and_mentions[0]
    mentions = relation_and_mentions[1]
    print(relation)
    print("Mentions: " + str(mentions))
    path_2_count = rel_2_paths_counts[relation]
    for path in path_2_count:
        print("\t" + path + ":" + str(path_2_count[path]))
    print()