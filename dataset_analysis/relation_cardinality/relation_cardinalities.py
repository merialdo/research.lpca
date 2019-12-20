"""
This is an utility module to compute relation cardinalities.

for each relation r
    - for each entity h that happens to be head for that relation in at least one fact
        - consider all facts <h, r, _ > in training, validation or test set
        - so count all distinct entities that are heads for t via r

    - for each entity t that happens to be tail for that relation in at least one fact
        - consider all facts < _ , r, t > in training, validation or test set
        - so count all distinct entities that are heads for t via r
"""

def _add_to_relation_matches(relation_2_matches, triples):

    for head, relation, tail in triples:

        if relation not in relation_2_matches:
            relation_2_matches[relation] = dict()
            relation_2_matches[relation]["head_to_tails"] = dict()
            relation_2_matches[relation]["tail_to_heads"] = dict()

        if head in relation_2_matches[relation]["head_to_tails"]:
            relation_2_matches[relation]["head_to_tails"][head].append(tail)
        else:
            relation_2_matches[relation]["head_to_tails"][head] = [tail]

        if tail in relation_2_matches[relation]["tail_to_heads"]:
            relation_2_matches[relation]["tail_to_heads"][tail].append(head)
        else:
            relation_2_matches[relation]["tail_to_heads"][tail] = [head]


def compute(dataset):
    """
        compute the cardinality for all the relations in a dataset
        based on training facts, validation facts, and test facts
    """
    #this is a dict of dicts
    # rel associates each relation to a dict()
    #       the dict has two keys: "head_to_tails" and "tail_to_heads"
    #              - rel_2_cardinality_dicts[rel]["head_to_tails"] is, in turn, a dict
    #                           that associates each head to the tails that it is connected to via rel
    #              - rel_2_cardinality_dicts[rel]["tail_to_heads"] is, in turn, a dict
    #                           that associates each tail to the heads that it is connected to via rel

    print("Computing relationship cardinalities for dataset %s" % dataset.name)
    rel_2_cardinality_dicts = dict()

    _add_to_relation_matches(rel_2_cardinality_dicts, dataset.train_triples)
    _add_to_relation_matches(rel_2_cardinality_dicts, dataset.valid_triples)
    _add_to_relation_matches(rel_2_cardinality_dicts, dataset.test_triples)

    return rel_2_cardinality_dicts
