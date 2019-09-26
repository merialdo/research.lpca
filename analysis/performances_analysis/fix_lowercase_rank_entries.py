import html

import datasets
import models
from performances import read_filtered_ranks_entries_for


datasets_names = [datasets.YAGO3_10]
for dataset_name in datasets_names:
    lower_filt_rank_entries = read_filtered_ranks_entries_for(models.CONVE, dataset_name)

    dataset = datasets.Dataset(dataset_name)

    lower_ent_name_2_proper_ent_name = dict()
    lower_rel_name_2_proper_rel_name = dict()

    for entity in dataset.entities:
        # in Yago3-10 there are both Turing_award and Turing_Award, but in our result files we only need Turing_Award.
        # This is ok because the models usually predict Turing_Award only
        if entity == "Turing_award":
            entity = "Turing_Award"
        lower_ent_name_2_proper_ent_name[html.unescape(entity.lower())] = entity

    for relationship in dataset.relationships:
        lower_rel_name_2_proper_rel_name[html.unescape(relationship.lower())] = relationship

    target_entries = []

    for entry in lower_filt_rank_entries:
        target_entry = dict()

        lower_head, lower_rel, lower_tail, lower_head_rank, lower_tail_rank = \
            entry['head'], entry['relation'], entry['tail'], int(entry['head_rank_filtered']), int(entry['tail_rank_filtered'])

        target_entry['head'] = lower_ent_name_2_proper_ent_name[lower_head]
        target_entry['relation'] = lower_rel_name_2_proper_rel_name[lower_rel]
        target_entry['tail'] = lower_ent_name_2_proper_ent_name[lower_tail]
        target_entry['head_rank_filtered'] = str(lower_head_rank)
        target_entry['tail_rank_filtered'] = str(lower_tail_rank)
        target_entries.append(target_entry)

    lines = []
    for x in target_entries:
        lines.append(";".join([x['head'], x['relation'], x['tail'], x['head_rank_filtered'], x['tail_rank_filtered']]) + "\n")


    with open("/Users/andrea/comparative_analysis/results/YAGO3-10/convE_filtered_ranks2.csv", "w") as outfile:
        outfile.writelines(lines)

