
# BASELINE
ANYBURL = "AnyBURL"


### TENSOR DECOMPOSITION ###
DISTMULT = "DistMult"
COMPLEX = "Complex"
ANALOGY = "ANALOGY"
SIMPLE = "SimplE"

HOLE = "HolE"
TUCKER = "TuckER"


### GEOMETRIC ###
TRANSE = "TransE"

STRANSE = "STransE"
CROSSE = "CrossE"
CROSSE_NS = "CrossE_NS"    # CrossE variant that does not use sigmoid in testing

TORUSE = "TorusE"
ROTATE = "RotatE"


### DEEP LEARNING ###
CONVE = "ConvE"
CONVKB = "ConvKB"
CONVR = "ConvR"

CAPSE = "CapsE"
CAPSE_LEAKY = "CapsE_leaky" # CapsE variant that uses Leaky ReLU instead of ReLU both in training and in testing

RSN = "RSN"


ALL_MODEL_NAMES = [ANYBURL, DISTMULT, COMPLEX, ANALOGY, SIMPLE, HOLE, TUCKER, TRANSE, STRANSE, CROSSE, TORUSE, ROTATE, CONVE, CONVKB, CONVR ,CAPSE, RSN]

RPS_MODEL = "RPS_model"