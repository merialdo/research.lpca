import html

import datasets
import models
from performances import read_filtered_ranks_entries_for, read_filtered_details_entries_for


datasets_names = [datasets.WN18]

for dataset_name in datasets_names:
    short_filt_det_entries = read_filtered_details_entries_for(models.RSN, dataset_name)

    dataset = datasets.Dataset(dataset_name)

    short_ent_name_2_proper_ent_name = dict()
    short_rel_name_2_proper_rel_name = dict()

    for entity in dataset.entities:
        short_entity = str(int(entity))
        short_ent_name_2_proper_ent_name[short_entity] = entity

    target_entries = []

    for entry in short_filt_det_entries:
        target_entry = dict()

        short_head, rel, short_tail, short_head_details, short_tail_details = \
            entry['head'], entry['relation'], entry['tail'], entry['head_details_filtered'], entry['tail_details_filtered']

        target_entry['head'] = short_ent_name_2_proper_ent_name[short_head]
        target_entry['relation'] = rel
        target_entry['tail'] = short_ent_name_2_proper_ent_name[short_tail]

        if '' in short_head_details:
            short_head_details.remove('')

        if '' in short_tail_details:
            short_tail_details.remove('')

        target_entry['head_details_filtered'] = [short_ent_name_2_proper_ent_name[x] for x in short_head_details]
        target_entry['tail_details_filtered'] = [short_ent_name_2_proper_ent_name[x] for x in short_tail_details]

        target_entries.append(target_entry)

    lines = []
    for x in target_entries:
        key = ";".join([x['head'], x['relation'], x['tail']])

        head_line = key + ";predict head;[" + ";".join(x['head_details_filtered']) + "]\n"
        tail_line = key + ";predict tail;[" + ";".join(x['tail_details_filtered']) + "]\n"
        lines.append(head_line)
        lines.append(tail_line)


    with open("/Users/andrea/comparative_analysis/results/WN18/rsn_filtered_details2.csv", "w") as outfile:
        outfile.writelines(lines)

