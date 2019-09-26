import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from bucket import bucket
from io_utils import *

def poly_func(x, coeffs):
    result = 0
    for i in range(len(coeffs)):
        coeff = coeffs[i]
        exp = len(coeffs) - 1 - i
        result += pow(x, exp) * coeff
    return result

def plot_distances_from(entries):
    degree_to_distances = defaultdict(lambda: [])
    for entry in entries:
        degree_to_distances[entry["degree"]].append(entry["distance"])

    matplotlib.rcParams.update({'font.size': 20})
    plt.rcParams.update({'font.size': 20})

    x = []
    y = []
    for item in sorted(degree_to_distances.items(), key=lambda item: item[0]):
        #for value in item[1]:
        #    x.append(item[0])
        #    y.append(value)
        x.append(item[0])
        y.append(np.average(item[1]))

    #coeffs = np.polyfit(x, y, 4)
    #y_polyfit = [poly_func(x, coeffs) for x in x]
    #plt.plot(x, y_polyfit, '--', label='NN avg distance approx.')
    #plt.legend(markerscale=2., scatterpoints=1, fontsize=14)


    bucket_x, bucket_y = bucket(x, y, 10)
    plt.scatter(bucket_x, bucket_y, marker='.', s=20, label='NN avg distance')
    #plt.legend(markerscale=2., scatterpoints=1, fontsize=14)


    plt.xscale('log')

    plt.xlabel("Entity degree")
    plt.ylabel("Closest neighbour distance")

    plt.grid(True)

    plt.show()


entries = get_entries_from_mid_degree_distance_file("/Users/andrea/paper/FB15K/transe_fb15k_degree_vs_distance.csv")
plot_distances_from(entries)