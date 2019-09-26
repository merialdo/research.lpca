from collections import defaultdict
from dataset_analysis.siblings import sibling_classes
from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10

sibling_class_2_facts = sibling_classes.read(YAGO3_10)

class_2_number_of_all_predictions = defaultdict(lambda :0)
all = 0
for fine_class in sibling_class_2_facts:
    test_facts = sibling_class_2_facts[fine_class]
    class_2_number_of_all_predictions[fine_class] += len(test_facts)

    all += len(test_facts)

percs = []
for fine_class in class_2_number_of_all_predictions:
    predictions = class_2_number_of_all_predictions[fine_class]
    perc = 100*float(predictions)/float(all)
    perc = round(perc, 2)
    percs.append(perc)
    print(fine_class)
    print(str(perc) + "%")
    print()
