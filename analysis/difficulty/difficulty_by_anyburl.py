import numpy as np
import matplotlib.pyplot as plt
from io_utils import *

def compute_difficulties(entries):

    prediction_2_difficulty_dict = dict()
    for entry in entries:
        prediction = [entry['head'], entry['relation'], entry['tail'], "predict head"]
        key = ";".join(prediction)
        difficulty = entry['head_rank_filtered']
        prediction_2_difficulty_dict[key] = difficulty

        prediction = [entry['head'], entry['relation'], entry['tail'], "predict tail"]
        key = ";".join(prediction)
        difficulty = entry['tail_rank_filtered']
        prediction_2_difficulty_dict[key] = difficulty

    return prediction_2_difficulty_dict



anyburl_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/anyburl_filtered_ranks.csv")


prediction_2_difficulty = compute_difficulties(anyburl_entries)
lines = []
for key in prediction_2_difficulty:
    lines.append(key + ";" + str(prediction_2_difficulty[key]) + "\n")


with open('difficulty_anyburl.csv', "w") as difficulty_file:
    difficulty_file.writelines(lines)