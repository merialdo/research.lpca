import matplotlib.pyplot as plt
import numpy as np

from datasets import FB15K, FB15K_237, WN18, WN18RR, YAGO3_10
from models import ALL_MODEL_NAMES
from efficiency.prediction_times import PREDICTION_TIMES

labels = ALL_MODEL_NAMES

prediction_times_fb15k = [PREDICTION_TIMES[FB15K][model_name]*1000 for model_name in ALL_MODEL_NAMES]
prediction_times_fb15k237 = [PREDICTION_TIMES[FB15K_237][model_name]*1000  for model_name in ALL_MODEL_NAMES]
prediction_times_wn18 = [PREDICTION_TIMES[WN18][model_name]*1000  for model_name in ALL_MODEL_NAMES]
prediction_times_wn18rr = [PREDICTION_TIMES[WN18RR][model_name]*1000  for model_name in ALL_MODEL_NAMES]
prediction_times_yago310 = [PREDICTION_TIMES[YAGO3_10][model_name]*1000  for model_name in ALL_MODEL_NAMES]


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 2*width, prediction_times_fb15k, width, label=FB15K, zorder=3)
rects2 = ax.bar(x - 1*width, prediction_times_fb15k237, width, label=FB15K_237, zorder=3)
rects3 = ax.bar(x, prediction_times_wn18, width, label=WN18, zorder=3)
rects4 = ax.bar(x + 1*width, prediction_times_wn18rr, width, label=WN18RR, zorder=3)
rects5 = ax.bar(x + 2*width, prediction_times_yago310, width, label=YAGO3_10, zorder=3)

FB15K_COLOR="#ED553B"
FB15K237_COLOR="#e8c441"
WN18_COLOR="#3CAEA3"
WN18RR_COLOR="#20639B"
YAGO310_COLOR="#173F5F"


for i in range(len(rects1)):
    rects1[i].set_color(FB15K_COLOR)
    rects2[i].set_color(FB15K237_COLOR)
    rects3[i].set_color(WN18_COLOR)
    rects4[i].set_color(WN18RR_COLOR)
    rects5[i].set_color(YAGO310_COLOR)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)

ax.set_ylabel('Prediction times (ms)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='vertical')
ax.grid(zorder=0)
ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=12)

ax.set_yscale('log')

fig.tight_layout()

plt.show()
