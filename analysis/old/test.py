from collections import defaultdict

from io_utils import get_entries_from_file

transe_entries = get_entries_from_file("/Users/andrea/paper/FB15K/distmult_fb15k_test_with_correct_ranks.csv")

entity_2_h10_count = dict()
for entry in transe_entries:
    entity_2_h10_count[entry["head"]] = 0
    entity_2_h10_count[entry["tail"]] = 0


for entry in transe_entries:
    if entry["head_rank_raw"] == 1:
        entity_2_h10_count[entry["head"]] += 1

    if entry["tail_rank_raw"] == 1:
        entity_2_h10_count[entry["tail"]] += 1

zeroes = 0
for entity in entity_2_h10_count:
    if entity_2_h10_count[entity] == 0:
        zeroes += 1
print(zeroes)