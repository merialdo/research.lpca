import html

import datasets
import models
from performances import read_filtered_ranks_entries_for, read_filtered_details_entries_for

datasets_names = [datasets.YAGO3_10]
for dataset_name in datasets_names:
    lower_filt_det_entries = read_filtered_details_entries_for(models.CONVE, dataset_name)

    dataset = datasets.Dataset(dataset_name)

    lower_ent_name_2_proper_ent_name = dict()
    lower_rel_name_2_proper_rel_name = dict()

    for entity in dataset.entities:
        lower_ent_name_2_proper_ent_name[html.unescape(entity.lower())] = entity

    for relationship in dataset.relationships:
        lower_rel_name_2_proper_rel_name[html.unescape(relationship.lower())] = relationship

    target_entries = []

    for entry in lower_filt_det_entries:
        target_entry = dict()

        lower_head, lower_rel, lower_tail, lower_head_details, lower_tail_details = \
            entry['head'], entry['relation'], entry['tail'], entry['head_details_filtered'], entry['tail_details_filtered']

        target_entry['head'] = lower_ent_name_2_proper_ent_name[lower_head]
        target_entry['relation'] = lower_rel_name_2_proper_rel_name[lower_rel]
        target_entry['tail'] = lower_ent_name_2_proper_ent_name[lower_tail]

        if '' in lower_head_details:
            lower_head_details.remove('')

        if '' in lower_tail_details:
            lower_tail_details.remove('')
        target_entry['head_details_filtered'] = [lower_ent_name_2_proper_ent_name[x] for x in lower_head_details]
        target_entry['tail_details_filtered'] = [lower_ent_name_2_proper_ent_name[x] for x in lower_tail_details]

        target_entries.append(target_entry)

    lines = []
    for x in target_entries:
        key = ";".join([x['head'], x['relation'], x['tail']])

        head_line = key + ";predict head;[" + ";".join(x['head_details_filtered']) + "]\n"
        tail_line = key + ";predict tail;[" + ";".join(x['tail_details_filtered']) + "]\n"
        lines.append(head_line)
        lines.append(tail_line)


    with open("/Users/andrea/comparative_analysis/results/YAGO3-10/convE_filtered_details2.csv", "w") as outfile:
        outfile.writelines(lines)

