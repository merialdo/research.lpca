import math

import models
import performances
from dataset_analysis.relation_properties import relation_properties
from dataset_analysis.relation_properties.relation_properties import *
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, Dataset
from io_utils import *
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
PROPERTIES_TO_CONSIDER = [REFLEXIVE, IRREFLEXIVE, SYMMETRIC, ANTISYMMETRIC, TRANSITIVE]

def compute_mrr(ranks):
    return np.average([1.0/float(rank) for rank in ranks])

dataset_name = FB15K
dataset = Dataset(dataset_name)

relation_2_properties = relation_properties.read(dataset_name)
all_test_facts_number = float(len(dataset.test_triples))

relation_type_2_test_facts = defaultdict(lambda:[])

for test_fact in dataset.test_triples:
    types = relation_2_properties[test_fact[1]]

    for rel_type in types:
        relation_type_2_test_facts[rel_type].append(test_fact)
    if len(types) == 2:
        relation_type_2_test_facts["both"].append(test_fact)




# === count and plot the percentage of facts for each type of relations ===
relation_type_2_facts_count = defaultdict(lambda: 0.0)
relation_type_2_facts_percentage = defaultdict(lambda: 0.0)

for relation_type in PROPERTIES_TO_CONSIDER:
    relation_type_2_facts_count[relation_type] = len(relation_type_2_test_facts[relation_type])
    relation_type_2_facts_percentage[relation_type] = float(relation_type_2_facts_count[relation_type])/float(len(dataset.test_triples))

# compute the x and y ticks that we will use in all our heatmaps
heatmap_x_ticks = ["any"] + [x.replace(" ", "\n") for x in PROPERTIES_TO_CONSIDER]
heatmap_y_ticks = []



models_for_dataset = models.ALL_MODEL_NAMES

for mod in models_for_dataset:
    heatmap_y_ticks.append(mod)

percentages_matrix = np.zeros(shape=(1, len(PROPERTIES_TO_CONSIDER)+1), dtype=np.float)

percentages_matrix[0, 0] = 1.0

for i in range(len(PROPERTIES_TO_CONSIDER)):
    percentages_matrix[0, 1+i] = relation_type_2_facts_percentage[PROPERTIES_TO_CONSIDER[i]]


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(18, 1)
ax1 = fig.add_subplot(gs[0:16, :])
ax2 = fig.add_subplot(gs[16, :])



ax = sns.heatmap(percentages_matrix,
            linewidth=0.5,
            annot=True,
            square=True,
            xticklabels=heatmap_x_ticks,
            yticklabels=[],
            cmap="OrRd",
            vmin=0.0,
            vmax=1.0,
            ax=ax2)
ax.set(xlabel="Relation type")

# ----------------------------------------------------------------------------------------------------

mrr_matrix = np.zeros(shape=[len(models_for_dataset), len(PROPERTIES_TO_CONSIDER)+1], dtype=np.float)

for i in range(len(models_for_dataset)):
    model_name = models_for_dataset[i]

    # === count the MRR for each source peers interval ===

    source_peer_interval_2_ranks_list = defaultdict(lambda: [])
    model_entries = performances.read_filtered_ranks_entries_for(model_name, dataset_name, entity_number=len(dataset.entities))

    relation_property_2_ranks = defaultdict(lambda: [])
    for entry in model_entries:
        head, relation, tail = entry['head'], entry['relation'], entry['tail']

        properties = relation_2_properties[relation]
        relation_property_2_ranks["any"].append(entry["head_rank_filtered"])
        relation_property_2_ranks["any"].append(entry["tail_rank_filtered"])

        for rel_property in properties:
            if relation_property_2_ranks == '':
                continue
            relation_property_2_ranks[rel_property].append(entry["head_rank_filtered"])
            relation_property_2_ranks[rel_property].append(entry["tail_rank_filtered"])

    rel_property_2_mrr = dict()

    for rel_property in relation_property_2_ranks:
        ranks = relation_property_2_ranks[rel_property]
        ranks = np.array(ranks)
        rel_property_2_mrr[rel_property] = compute_mrr(ranks)

    mrr_matrix[i, 0] = round(rel_property_2_mrr["any"], 2) if "any" in rel_property_2_mrr else None
    for j in range(len(PROPERTIES_TO_CONSIDER)):
        mrr_matrix[i, j+1] = round(rel_property_2_mrr[PROPERTIES_TO_CONSIDER[j]], 2) if PROPERTIES_TO_CONSIDER[j] in rel_property_2_mrr else None


print(mrr_matrix)
sns.heatmap(mrr_matrix,
                linewidth=0.5,
                annot=True,
                square=True,
                xticklabels=[],
                yticklabels=heatmap_y_ticks,
                cmap="coolwarm_r",
                vmin=0.0,
                vmax=1.0,
                ax=ax1)
plt.show()

