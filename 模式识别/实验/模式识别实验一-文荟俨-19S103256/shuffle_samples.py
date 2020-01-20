import random
import numpy as np
with open("./samples.txt", "r") as f:
    data = f.readlines()
index = [i for i in range(0, 1500)]
random.shuffle(index)
for i in range(0, len(data)):
    with open("./samples2.txt", "a") as f2:
        f2.write(data[index[i]])
