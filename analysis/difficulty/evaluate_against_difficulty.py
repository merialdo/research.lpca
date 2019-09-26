import numpy as np
import matplotlib.pyplot as plt

from dataset_analysis.degrees import entity_degrees, relation_mentions
from dataset_analysis.relation_cardinalities import relation_coarse_classes
from io_utils import *

rotate_entries = get_entries_from_file("/Users/andrea/comparative_analysis/results/FB15k/rotatE_filtered_ranks.csv")
difficulties = get_difficulties_from_file(
    "/Users/andrea/comparative_analysis/analysis/difficulty/difficulty_anyburl.csv")

_, _, entity2degree = entity_degrees.read("/Users/andrea/comparative_analysis/datasets/FB15k/entity_degrees.csv")
relation2mentions = relation_mentions.read(
    "/Users/andrea/comparative_analysis/datasets/FB15k/relation_mentions.csv")
relation_2_type = relation_coarse_classes.read("/Users/andrea/comparative_analysis/datasets/FB15k/relation_types.csv")


def analyze_overall_relations_for_predictions(predictions):
    overall_one_2_one_count = 0.0
    overall_many_2_one_count = 0.0
    overall_one_2_many_count = 0.0
    overall_many_2_many_count = 0.0
    overall_all = 0.0
    for prediction in predictions:
        head, relation, tail, type = prediction.split(";")

        overall_all += 1
        if relation_2_type[relation] == "one to one":
            overall_one_2_one_count += 1
        elif relation_2_type[relation] == "one to many":
            overall_one_2_many_count += 1
        elif relation_2_type[relation] == "many to one":
            overall_many_2_one_count += 1
        else:
            overall_many_2_many_count += 1

    overall_one_2_one_percentage = float(overall_one_2_one_count)*100/float(overall_all)
    overall_many_2_one_percentage = float(overall_many_2_one_count)*100/float(overall_all)
    overall_one_2_many_percentage = float(overall_one_2_many_count)*100/float(overall_all)
    overall_many_2_many_percentage = float(overall_many_2_many_count)*100/float(overall_all)
    print("Overall relation types ratios in test set")
    print("\tOne to one: %f%%; One to many: %f%%; Many to one: %f%%; Many to many: %f%%;" %
          (overall_one_2_one_percentage, overall_many_2_one_percentage, overall_one_2_many_percentage, overall_many_2_many_percentage))
    print()

def get_differences(model_entries, baseline_difficulties):

    model_baseline_both_hit = []
    model_hit__baseline_miss = []
    model_miss__baseline_hit = []
    model_baseline_both_miss = []

    for entry in model_entries:
        prediction = [entry['head'], entry['relation'], entry['tail'], "predict head"]
        key = ";".join(prediction)

        if float(entry['head_rank_filtered']) == 1.0:
            if baseline_difficulties[key] == 1.0:
                model_baseline_both_hit.append(key)
            else:
                model_hit__baseline_miss.append(key)
        else:
            if baseline_difficulties[key] == 1.0:
                model_miss__baseline_hit.append(key)
            else:
                model_baseline_both_miss.append(key)

        prediction = [entry['head'], entry['relation'], entry['tail'], "predict tail"]
        key = ";".join(prediction)

        if float(entry['tail_rank_filtered']) == 1.0:
            if baseline_difficulties[key] == 1.0:
                model_baseline_both_hit.append(key)
            else:
                model_hit__baseline_miss.append(key)
        else:
            if baseline_difficulties[key] == 1.0:
                model_miss__baseline_hit.append(key)
            else:
                model_baseline_both_miss.append(key)


    return model_baseline_both_hit, model_hit__baseline_miss, model_miss__baseline_hit, model_baseline_both_miss

def analyze_predictions(predictions):
    print("\tPredictions: " + str(len(predictions)))
    degrees_of_entities_to_predict = []

    one_2_one_count = 0.0
    many_2_one_count = 0.0
    one_2_many_count = 0.0
    many_2_many_count = 0.0
    all_count = 0.0

    for prediction in predictions:
        head, relation, tail, prediction_type = prediction.split(";")
        entity_to_predict = head if prediction_type == "predict head" else tail
        degrees_of_entities_to_predict.append(entity2degree[entity_to_predict])
        all_count += 1
        if relation_2_type[relation] == "one to one":
            one_2_one_count += 1
        elif relation_2_type[relation] == "one to many":
            one_2_many_count += 1
        elif relation_2_type[relation] == "many to one":
            many_2_one_count += 1
        else:
            many_2_many_count += 1

    one_2_one_percentage = one_2_one_count*100/all_count
    one_2_many_percentage = one_2_many_count*100/all_count
    many_2_one_percentage = many_2_one_count*100/all_count
    many_2_many_percentage = many_2_many_count*100/all_count

    print("\tAVG of the degree of the entity to predict: " + str(np.average(degrees_of_entities_to_predict)))
    print("\tSTD of the degree of the entity to predict: " + str(np.std(degrees_of_entities_to_predict)))
    print("\tOne to one: %f%%; One to many: %f%%; Many to one: %f%%; Many to many: %f%%;" %
          (one_2_one_percentage, one_2_many_percentage, many_2_one_percentage, many_2_many_percentage))

analyze_overall_relations_for_predictions(difficulties.keys())

both_hit, rotate_hit__anyburl_miss, rotate_miss__anyburl_hit, both_miss = get_differences(rotate_entries, difficulties)

print("Both RotatE and AnyBURL Hit")
analyze_predictions(both_hit)
print()

print("RotatE Hits when AnyBURL Misses")
analyze_predictions(rotate_hit__anyburl_miss)
print()

print("RotatE Misses when AnyBURL Hits")
analyze_predictions(rotate_miss__anyburl_hit)
print()

print("Both RotatE and AnyBURL Miss")
analyze_predictions(both_miss)
print()
