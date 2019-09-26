from io_utils import *
import numpy as np

def append_difficulties(entries, difficulties_dict):

    for entry in entries:
        prediction = [entry['head'], entry['relation'], entry['tail'], "predict head"]
        key = ";".join(prediction)
        difficulty = entry['head_rank_filtered']
        difficulties_dict[key].append(float(difficulty))

        prediction = [entry['head'], entry['relation'], entry['tail'], "predict tail"]
        key = ";".join(prediction)
        difficulty = entry['tail_rank_filtered']
        difficulties_dict[key].append(float(difficulty))

transE_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/transE_filtered_ranks.csv")
rotatE_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/rotatE_filtered_ranks.csv")
convE_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/convE_filtered_ranks.csv")
simplE_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/simplE_filtered_ranks.csv")
anyburl_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/AnyBURL_filtered_ranks.csv")

prediction_2_difficulty = defaultdict(lambda: [])

append_difficulties(transE_entries, prediction_2_difficulty)
append_difficulties(rotatE_entries, prediction_2_difficulty)
append_difficulties(convE_entries, prediction_2_difficulty)
append_difficulties(simplE_entries, prediction_2_difficulty)
append_difficulties(anyburl_entries, prediction_2_difficulty)



for key in prediction_2_difficulty:
    print(key + ": " + str(prediction_2_difficulty[key]))
    prediction_2_difficulty[key] = np.average(prediction_2_difficulty[key])


lines = []
for key in prediction_2_difficulty:
    lines.append(key + ";" + str(prediction_2_difficulty[key]) + "\n")

with open('difficulty_averaging.csv', "w") as difficulty_file:
    difficulty_file.writelines(lines)