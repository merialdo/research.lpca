from dataset_analysis.degrees import degree_classes
from dataset_analysis.degrees.degree_classes import CLASSES, INTERVALS
from datasets import FB15K
from io_utils import *
from models import SIMPLE, TRANSE, ROTATE, CONVE, RSN, TUCKER, ANYBURL
from performances import read_filtered_ranks_entries_for

model_entries = read_filtered_ranks_entries_for(ANYBURL, FB15K)

test_fact_2_class = degree_classes.read(FB15K, return_fact_2_class=True)

overall_all = 0.0

# build a dict that, for each degree class, tells us how many test facts belong to that degree class
degree_class_2_overall_counts = dict()
for degree_class in CLASSES:
    degree_class_2_overall_counts[degree_class] = 0.0
for entry in model_entries:
    head, relation, tail = entry['head'], entry['relation'], entry['tail']
    degree_class = test_fact_2_class[";".join([head, relation, tail])]
    overall_all += 1
    degree_class_2_overall_counts[degree_class] += 1

# these will be used to generate the CSV data
degree_class_2_head_hits_percentage = dict()
degree_class_2_tail_hits_percentage = dict()
degree_class_2_head_misses_percentage = dict()
degree_class_2_tail_misses_percentage = dict()

# compute
# - the number of all head hits@1 in the entire test set
# - the number of all head miss@1 in the entire test set

# - the number of all tail hits@1 in the entire test set
# - the number of all tail miss@1 in the entire test set

# - the number of head hits@1 specific for each degree class
# - the number of head miss@1 specific for each degree class

# - the number of tail hits@1 specific for each degree class
# - the number of tail miss@1 specific for each degree class

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

    degree_class = test_fact_2_class[";".join([head, relation, tail])]

    if head_rank_filtered == 1:
        all_head_hits+=1
        head_hits[degree_class] += 1
    else:
        all_head_misses+=1
        head_misses[degree_class] += 1

    if tail_rank_filtered == 1:
        all_tail_hits+=1
        tail_hits[degree_class] += 1
    else:
        all_tail_misses+=1
        tail_misses[degree_class] += 1


print("All head predictions: " + str(all_head_hits + all_head_misses))
print()

print("All head hits: " + str(all_head_hits))

for degree_class in CLASSES:
    if degree_class_2_overall_counts[degree_class] > 0:
        perc = float(head_hits[degree_class])*100/degree_class_2_overall_counts[degree_class]
        perc = round(perc, 2)
        print("\t" + degree_class + ": " + str(perc) + "%% of all " + degree_class + " head questions")
        # this will be useful for the CSV generation
        degree_class_2_head_hits_percentage[degree_class] = perc
    else:
        print("\t" + degree_class + ": " + " has no facts in the test set")
        degree_class_2_head_hits_percentage[degree_class] = "--"
print()

print("All head misses: " + str(all_head_misses))
all_head_miss_percs = []
for degree_class in CLASSES:
    if degree_class_2_overall_counts[degree_class] > 0:
        perc = float(head_misses[degree_class])*100/degree_class_2_overall_counts[degree_class]
        perc = round(perc, 2)
        print("\t" + degree_class + ": " + str(perc) + "%% of all " + degree_class + " head questions")
        # this will be useful for the CSV generation
        degree_class_2_head_misses_percentage[degree_class] = perc
    else:
        print("\t" + degree_class + ": " + " has no facts in the test set")
        degree_class_2_head_misses_percentage[degree_class] = "--"
print("\n")

print("All tail predictions: " + str(all_tail_hits + all_tail_misses))
print()

print("All tail hits: " + str(all_tail_hits))
all_tail_hit_percs = []
for degree_class in CLASSES:

    if degree_class_2_overall_counts[degree_class] > 0:
        perc = float(tail_hits[degree_class])*100/degree_class_2_overall_counts[degree_class]
        perc = round(perc, 2)
        print("\t" + degree_class + ": " + str(perc) + "%% of all " + degree_class + " tail questions")
        # this will be useful for the CSV generation
        degree_class_2_tail_hits_percentage[degree_class] = perc
    else:
        print("\t" + degree_class + ": " + " has no facts in the test set")
        degree_class_2_tail_hits_percentage[degree_class] = "--"
print()

print("All tail misses: " + str(all_tail_misses))
all_tail_miss_percs = []
for degree_class in CLASSES:

    if degree_class_2_overall_counts[degree_class] > 0:
        perc = float(tail_misses[degree_class])*100/degree_class_2_overall_counts[degree_class]
        perc = round(perc, 2)
        print("\t" + degree_class + ": " + str(perc) + "%% of all " + degree_class + " tail questions")
        # this will be useful for the CSV generation
        degree_class_2_tail_misses_percentage[degree_class] = perc
    else:
        print("\t" + degree_class + ": " + " has no facts in the test set")
        degree_class_2_tail_misses_percentage[degree_class] = "--"

print("\n\n\n")


print("Head prediction percentages for each interval, in CSV format")
for head_degree_interval in INTERVALS:
    row_percs = []
    for tail_degree_interval in INTERVALS:
        head_degree_interval_lower_bound = str(head_degree_interval[0])
        head_degree_interval_upper_bound = str(head_degree_interval[1])
        tail_degree_interval_lower_bound = str(tail_degree_interval[0])
        tail_degree_interval_upper_bound = str(tail_degree_interval[1])
        degree_class = head_degree_interval_lower_bound + "-" + head_degree_interval_upper_bound + "__" + tail_degree_interval_lower_bound + "-" + tail_degree_interval_upper_bound

        row_percs.append(str(degree_class_2_head_hits_percentage[degree_class]))

    print("; ".join(row_percs).replace('.', ','))


print()

print("Tail prediction percentages for each interval, in CSV format")

for head_degree_interval in INTERVALS:
    row_percs = []

    for tail_degree_interval in INTERVALS:
        head_degree_interval_lower_bound = str(head_degree_interval[0])
        head_degree_interval_upper_bound = str(head_degree_interval[1])
        tail_degree_interval_lower_bound = str(tail_degree_interval[0])
        tail_degree_interval_upper_bound = str(tail_degree_interval[1])
        degree_class = head_degree_interval_lower_bound + "-" + head_degree_interval_upper_bound + "__" + tail_degree_interval_lower_bound + "-" + tail_degree_interval_upper_bound
        row_percs.append(str(degree_class_2_tail_hits_percentage[degree_class]))
    print("; ".join(row_percs).replace('.', ','))
