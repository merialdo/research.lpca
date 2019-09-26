from dataset_analysis.peers.peer_classes import PEER_CLASSES, PEER_INTERVALS
from dataset_analysis.peers import peer_classes
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10
from io_utils import *
from models import ANYBURL, TRANSE, CONVE, TUCKER, ROTATE, SIMPLE, RSN, CONVR, CROSSE
from performances import read_filtered_ranks_entries_for

dataset_name = WN18RR
model_entries = read_filtered_ranks_entries_for(CONVR, dataset_name)

test_fact_2_class = peer_classes.read(dataset_name, return_fact_2_class=True)

# === count and print the percentage of facts in each peer class ===
overall_all = 0.0
peers_class_2_overall_counts = dict()
peers_class_2_percentage = dict()
# initialize the data structure
for peer_class in PEER_CLASSES:
    peers_class_2_overall_counts[peer_class] = 0.0

# count the occurrences for each peer class and overall
for entry in model_entries:
    head, relation, tail = entry['head'], entry['relation'], entry['tail']
    peer_class = test_fact_2_class[";".join([head, relation, tail])]
    overall_all += 1
    peers_class_2_overall_counts[peer_class] += 1
# compute the percentage for each peer class
for peer_class in peers_class_2_overall_counts:
    peers_class_2_percentage[peer_class] = float(peers_class_2_overall_counts[peer_class]) / float(overall_all)

# print the percentage of facts in each peer class
print("Overall peer class ratios in test set")
rows = []
for head_sib_interval in PEER_INTERVALS:
    row_bits = []
    for tail_sib_interval in PEER_INTERVALS:
        peer_class = str(head_sib_interval[0]) + "-" + str(head_sib_interval[1]) + "__" + str(
            tail_sib_interval[0]) + "-" + str(tail_sib_interval[1])
        row_bits.append(str(round(100 * peers_class_2_percentage[peer_class], 2)))
    rows.append(";".join(row_bits).replace("." , ","))
for row in rows:
    print(row)

# === count and print the percentage of hits@1 for each peer class ===

all_head_hits = 0
head_hits = defaultdict(lambda: 0)
all_head_misses = 0
head_misses = defaultdict(lambda: 0)

all_tail_hits = 0
tail_hits = defaultdict(lambda: 0)
all_tail_misses = 0
tail_misses = defaultdict(lambda: 0)

for entry in model_entries:

    head = entry['head']
    relation = entry['relation']
    tail = entry['tail']
    head_rank_filtered = entry['head_rank_filtered']
    tail_rank_filtered = entry['tail_rank_filtered']

    peer_class = test_fact_2_class[";".join([head, relation, tail])]

    if head_rank_filtered == 1:
        all_head_hits += 1
        head_hits[peer_class] += 1
    else:
        all_head_misses += 1
        head_misses[peer_class] += 1

    if tail_rank_filtered == 1:
        all_tail_hits += 1
        tail_hits[peer_class] += 1
    else:
        all_tail_misses += 1
        tail_misses[peer_class] += 1

print("All head predictions: " + str(all_head_hits + all_head_misses))
print()

print("All head hits: " + str(all_head_hits))
all_head_hit_percs = []
for head_peers_interval in PEER_INTERVALS:
    row_percs = []

    for tail_peers_interval in PEER_INTERVALS:
        head_peers_interval_lower_bound = str(head_peers_interval[0])
        head_peers_interval_upper_bound = str(head_peers_interval[1])
        tail_peers_interval_lower_bound = str(tail_peers_interval[0])
        tail_peers_interval_upper_bound = str(tail_peers_interval[1])
        fine_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

        if peers_class_2_overall_counts[fine_class] > 0:
            perc = float(head_hits[fine_class]) * 100 / peers_class_2_overall_counts[fine_class]
            perc = round(perc, 2)
            print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")
            row_percs.append(str(perc))
        else:
            row_percs.append('--')

    all_head_hit_percs.append(row_percs)

print()
print("All head misses: " + str(all_head_misses))
all_head_miss_percs = []
for head_peers_interval in PEER_INTERVALS:
    row_percs = []

    for tail_peers_interval in PEER_INTERVALS:
        head_peers_interval_lower_bound = str(head_peers_interval[0])
        head_peers_interval_upper_bound = str(head_peers_interval[1])
        tail_peers_interval_lower_bound = str(tail_peers_interval[0])
        tail_peers_interval_upper_bound = str(tail_peers_interval[1])
        fine_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

        if peers_class_2_overall_counts[fine_class] > 0:
            perc = float(head_misses[fine_class]) * 100 / peers_class_2_overall_counts[fine_class]
            perc = round(perc, 2)
            print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " head questions")
            row_percs.append(str(perc))
        else:
            row_percs.append('--')

    all_head_miss_percs.append(row_percs)

print()
print()

print("All tail predictions: " + str(all_tail_hits + all_tail_misses))
print()

print("All tail hits: " + str(all_tail_hits))
all_tail_hit_percs = []
for head_peers_interval in PEER_INTERVALS:
    row_percs = []

    for tail_peers_interval in PEER_INTERVALS:
        head_peers_interval_lower_bound = str(head_peers_interval[0])
        head_peers_interval_upper_bound = str(head_peers_interval[1])
        tail_peers_interval_lower_bound = str(tail_peers_interval[0])
        tail_peers_interval_upper_bound = str(tail_peers_interval[1])
        fine_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

        if peers_class_2_overall_counts[fine_class] > 0:
            perc = float(tail_hits[fine_class]) * 100 / peers_class_2_overall_counts[fine_class]
            perc = round(perc, 2)
            print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " tail questions")
            row_percs.append(str(perc))
        else:
            row_percs.append('--')

    all_tail_hit_percs.append(row_percs)

print()
print("All tail misses: " + str(all_tail_misses))
all_tail_miss_percs = []
for head_peers_interval in PEER_INTERVALS:
    row_percs = []

    for tail_peers_interval in PEER_INTERVALS:
        head_peers_interval_lower_bound = str(head_peers_interval[0])
        head_peers_interval_upper_bound = str(head_peers_interval[1])
        tail_peers_interval_lower_bound = str(tail_peers_interval[0])
        tail_peers_interval_upper_bound = str(tail_peers_interval[1])
        fine_class = head_peers_interval_lower_bound + "-" + head_peers_interval_upper_bound + "__" + tail_peers_interval_lower_bound + "-" + tail_peers_interval_upper_bound

        if peers_class_2_overall_counts[fine_class] > 0:
            perc = float(tail_misses[fine_class]) * 100 / peers_class_2_overall_counts[fine_class]
            perc = round(perc, 2)
            print("\t" + fine_class + ": " + str(perc) + "%% of all " + fine_class + " tail questions")
            row_percs.append(str(perc))
        else:
            row_percs.append('--')

    all_tail_miss_percs.append(row_percs)

print("\n\n\n")
for x in all_head_hit_percs:
    print("; ".join(x).replace('.', ','))

print()

for x in all_tail_hit_percs:
    print("; ".join(x).replace('.', ','))
