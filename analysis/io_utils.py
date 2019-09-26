from collections import defaultdict

def get_difficulties_from_file(filepath):
    predictions = dict()
    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            (head, relation, tail, prediction_type, value) = line.strip().split(";")
            predictions[";".join([head, relation, tail, prediction_type])] = float(value)

    return predictions

def get_entries_from_mid_degree_distance_file(filepath):
    entries = []
    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            entry = dict()

            (mid, degree, distance) = line.strip().split(";")
            entry["mid"] = mid
            entry["degree"] = int(degree)
            entry["distance"] = float(distance)

            entries.append(entry)
    return entries


def get_degrees_from_openke_file(filepath):
    entity_2_degree = defaultdict(lambda:0)
    relation_2_mentions = defaultdict(lambda:0)

    with open(filepath) as input_data:
        lines = input_data.readlines()[1:]
        for line in lines:
            (head, tail, relation) = line.strip().split(" ")

            head = int(head.strip())
            tail = int(tail.strip())
            relation = int(relation.strip())

            entity_2_degree[head] += 1
            entity_2_degree[tail] += 1
            relation_2_mentions[relation] += 1

    return entity_2_degree, relation_2_mentions


def get_degrees_from_openke_files(filepaths):
    entity_2_degree = defaultdict(lambda: 0)
    relation_2_mentions = defaultdict(lambda: 0)

    for filepath in filepaths:
        print(filepath)
        with open(filepath) as input_data:
            lines = input_data.readlines()[1:]
            for line in lines:
                (head, tail, relation) = line.strip().split(" ")

                head = int(head.strip())
                tail = int(tail.strip())
                relation = int(relation.strip())

                entity_2_degree[head] += 1
                entity_2_degree[tail] += 1
                relation_2_mentions[relation] += 1

    return entity_2_degree, relation_2_mentions


def get_entries_from_progressively_poorer_entities_from_file(filepath):
    entries = []

    with open(filepath) as input_data:
        lines = input_data.readlines()
        for line in lines:
            (skipped_entities, meanrank, h10) = line.strip().split(";")
            skipped_entities = int(skipped_entities.strip())
            meanrank = float(meanrank.strip())
            h10 = float(h10.strip())

            entry = dict()
            entry["k"] = skipped_entities
            entry["mean_rank"] = meanrank
            entry["h10"] = h10

            entries.append(entry)

    return entries


def get_facts_from_openke_facts_files(facts_files):
    facts = []

    for facts_file in facts_files:
        with open(facts_file) as input_data:
            lines = input_data.readlines()
            for line in lines[1:]:
                (head_id, tail_id, rel_id) = line.strip().split(" ")

                fact = dict()
                fact["head"] = head_id
                fact["tail"] = tail_id
                fact["relation"] = rel_id

                facts.append(fact)

    return facts
