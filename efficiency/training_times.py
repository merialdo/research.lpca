from datasets import YAGO3_10, WN18RR, WN18, FB15K_237, FB15K
from models import *

# TIMES ARE IN HOURS

TRAINING_TIMES_FB15K = {
                DISTMULT: 2,
                COMPLEX: 16,
                ANALOGY: 2,
                SIMPLE: 8,
                HOLE: 4,
                TUCKER: 3,
                TRANSE: 3,
                STRANSE: 13.5,
                CROSSE: 38,
                TORUSE: 3,
                ROTATE: 8,
                CONVE: 61,
                CONVKB: 1.5,
                CONVR: 16,
                CAPSE: 5.5,
                RSN: 4,
                ANYBURL: 0.03
                }

TRAINING_TIMES_WN18 = {
                DISTMULT: 1,
                COMPLEX: 1,
                ANALOGY: 0.5,
                SIMPLE: 1,
                HOLE: 2,
                TUCKER: 2,
                TRANSE: 2,
                STRANSE: 1.5,
                CROSSE: 19,
                TORUSE: 1,
                ROTATE: 5,
                CONVE: 16,
                CONVKB: 0.375,
                CONVR: 12,
                CAPSE: 0.185,
                RSN: 2,
                ANYBURL: 0.03
}

TRAINING_TIMES_FB15K_237 = {
                DISTMULT: 3,
                COMPLEX: 4,
                ANALOGY: 0.66,
                SIMPLE: 3,
                HOLE: 6,
                TUCKER: 2,
                TRANSE: 1,
                STRANSE: 7.5,
                CROSSE: 27,
                TORUSE: 2,
                ROTATE: 5,
                CONVE: 32,
                CONVKB: 0.65,
                CONVR: 15,
                CAPSE: 0.175,
                RSN: 3,
                ANYBURL: 0.28
}

TRAINING_TIMES_WN18RR = {
                DISTMULT: 2,
                COMPLEX: 4,
                ANALOGY: 0.25,
                SIMPLE: 1,
                HOLE: 0.5,
                TUCKER: 2,
                TRANSE: 1,
                STRANSE: 0.5,
                CROSSE: 15,
                TORUSE: 0.5,
                ROTATE: 4,
                CONVE: 10,
                CONVKB: 0.25,
                CONVR: 12,
                CAPSE: 0.185,
                RSN: 2,
                ANYBURL: 0.28
}

TRAINING_TIMES_YAGO3_10 = {
                DISTMULT: 20,
                COMPLEX: 30,
                ANALOGY: 1,
                SIMPLE: 20,
                HOLE: 24,
                TUCKER: 16,
                TRANSE: 10,
                STRANSE: 100,
                CROSSE: 280,
                TORUSE: 6,
                ROTATE: 5,
                CONVE: 200,
                CONVKB: 3,
                CONVR: 200,
                CAPSE: 9,
                RSN: 3,
                ANYBURL: 0.28
}

TRAINING_TIMES = dict()
TRAINING_TIMES[FB15K] = TRAINING_TIMES_FB15K
TRAINING_TIMES[FB15K_237] = TRAINING_TIMES_FB15K_237
TRAINING_TIMES[WN18] = TRAINING_TIMES_WN18
TRAINING_TIMES[WN18RR] = TRAINING_TIMES_WN18RR
TRAINING_TIMES[YAGO3_10] = TRAINING_TIMES_YAGO3_10