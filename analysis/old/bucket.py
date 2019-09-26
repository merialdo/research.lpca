import numpy as np

def bucket(x_array, y_array, bucket_size):
    new_x_array = []
    new_y_array = []


    zipped = zip(x_array, y_array)
    zipped = sorted(zipped, key=lambda pair: pair[0])

    sorted_x_array = [x for x,_ in zipped]
    sorted_y_array = [y for _,y in zipped]

    start = 0

    while(start <= sorted_x_array[-1]):
        temp_x_array = []
        temp_y_array = []

        end = min(start + bucket_size, sorted_x_array[-1]+1)

        for i in range(len(sorted_x_array)):
            x = sorted_x_array[i]
            y = sorted_y_array[i]

            if start <= x < end:
                temp_x_array.append(x)
                temp_y_array.append(y)

            if x > end:
                break

        start = end
        if len(temp_x_array) > 0:
            new_x_array.append(np.average(temp_x_array))
            new_y_array.append(np.average(temp_y_array))

    return new_x_array, new_y_array
