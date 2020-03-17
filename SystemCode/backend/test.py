import pickle
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt


def seriesToDataFrame(recorded_data):
    m = []
    index = []
    columns = [l[0] for l in recorded_data[-1]]
    for k, v in recorded_data.items():
        if (type(v) == list):
            m.append(list(zip(*v))[1])
            # m.append((v[0][1], v[1][1], v[2][1], v[3][1]))
            index.append(k)

    df = pd.DataFrame(m, columns=columns)
    df.index = index  # by right, can just use allocation.index, but there are some NaN values
    return df


allocation = pickle.load(open("data.pickle", "rb"))
print()

# m = []
# index = []
# for k, v in allocation.items():
#     m.append((v[0][1], v[1][1], v[2][1], v[3][1]))
#     index.append(k)

# df = pd.DataFrame(m, columns=['VTI', 'VXUS', 'BND', 'BNDX'])
# df.index = index  # by right, can just use allocation.index, but there are some NaN values
df = seriesToDataFrame(allocation)
df.plot.area()
plt.show()
print()
