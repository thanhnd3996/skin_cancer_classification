# from tqdm import tqdm
#
# n_classes = 7
# for class_i in tqdm(range(n_classes), miniters=1):
#     print(class_i)


import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.concatenate((a, b), axis=0)
print(c)

np.save('a.npy', a)