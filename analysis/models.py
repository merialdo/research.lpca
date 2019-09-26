import os

TRANSE = "transE"
DISTMULT = "distmult"
HOLE = "holE"
COMPLEX = "complex"
SIMPLE = "simplE"
CONVE = "convE"
ROTATE = "rotatE"
RSN = "rsn"
TUCKER = "tuckER"
ANYBURL = "anyBURL"
CROSSE = "crossE"
CONVR = "convR"
ANALOGY = "analogy"

ALL_MODEL_NAMES = [TRANSE, DISTMULT, HOLE, COMPLEX, CONVE, SIMPLE, CROSSE, ANALOGY, CONVR, ROTATE, RSN, TUCKER, ANYBURL]

HOME = "/Users/andrea/comparative_analysis/"
def filtered_ranks_path(model_name, dataset_name):
    path = os.path.join(HOME, "results", dataset_name, model_name + "_filtered_ranks.csv")
    return os.path.abspath(path)

def filtered_details_path(model_name, dataset_name):
    path = os.path.join(HOME, "results", dataset_name, model_name + "_filtered_details.csv")
    return os.path.abspath(path)

def get_models_supporting_dataset(dataset_name):
    result = []
    for model_name in ALL_MODEL_NAMES:
        if os.path.isfile(os.path.join(HOME, "results", dataset_name, model_name + "_filtered_ranks.csv")):
            result.append(model_name)
    return result