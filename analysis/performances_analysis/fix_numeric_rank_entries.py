import html

import datasets
import models
from performances import read_filtered_ranks_entries_for


datasets_names = [datasets.WN18]

for dataset_name in datasets_names:
    short_filt_rank_entries = read_filtered_ranks_entries_for(models.RSN, dataset_name)

    dataset = datasets.Dataset(dataset_name)

    short_ent_name_2_proper_ent_name = dict()
    short_rel_name_2_proper_rel_name = dict()

    for entity in dataset.entities:
        short_entity = str(int(entity))
        short_ent_name_2_proper_ent_name[short_entity] = entity

    target_entries = []

    for entry in short_filt_rank_entries:
        target_entry = dict()

        short_head, rel, short_tail, head_rank, tail_rank = \
            entry['head'], entry['relation'], entry['tail'], int(entry['head_rank_filtered']), int(entry['tail_rank_filtered'])

        target_entry['head'] = short_ent_name_2_proper_ent_name[short_head]
        target_entry['relation'] = rel
        target_entry['tail'] = short_ent_name_2_proper_ent_name[short_tail]
        target_entry['head_rank_filtered'] = str(head_rank)
        target_entry['tail_rank_filtered'] = str(tail_rank)
        target_entries.append(target_entry)

    lines = []
    for x in target_entries:
        lines.append(";".join([x['head'], x['relation'], x['tail'], x['head_rank_filtered'], x['tail_rank_filtered']]) + "\n")


    with open("/Users/andrea/comparative_analysis/results/WN18/RSN_filtered_ranks2.csv", "w") as outfile:
        outfile.writelines(lines)

