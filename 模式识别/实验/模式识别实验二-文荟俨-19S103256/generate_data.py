import numpy as np
import random


def generate_data(u, sigma, category, num_sample):
    sample = np.random.multivariate_normal(u, sigma, num_sample)
    data = []
    for x in sample:
        data.append([x[0], x[1], x[2], category])
    with open("samples.txt", "a") as f:
        for i in range(len(data)):
            f.write(str(data[i]) + '\n')


def init_data():
    # Init u and sigma
    u1 = np.asarray([0, 0, 0])
    sigma1 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    u2 = np.asarray([0, 1, 0])
    sigma2 = np.asarray([[1, 0, 1], [0, 2, 2], [1, 2, 5]])
    u3 = np.asarray([-1, 0, 1])
    sigma3 = np.asarray([[2, 0, 0], [0, 6, 0], [0, 0, 1]])
    u4 = np.asarray([0, 0.5, 1])
    sigma4 = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
    # Generate data
    num_sample = 1100
    generate_data(u1, sigma1, 0, num_sample)
    generate_data(u2, sigma2, 1, num_sample)
    generate_data(u3, sigma3, 2, num_sample)
    generate_data(u4, sigma4, 3, num_sample)


def shuffle_data():
    with open("train_samples.txt", "r") as f:
        data = f.readlines()
    random.shuffle(data)
    with open("samples_shuffle.txt", "a") as f:
        for x in data:
            f.write(str(x))


def divide_data():
    with open("samples.txt", "r") as f:
        data = f.readlines()
    f.close()
    for i in range(len(data)):
        if ((i >= 0) & (i <= 99)) | ((i >= 1100) & (i <= 1199)) | ((i >= 2200) & (i <= 2299)) | ((i >= 3300) & (i <= 3399)):
            with open("test_samples.txt", "a") as f:
                f.write(data[i])
                f.close()
        else:
            with open("train_samples.txt", "a") as f:
                f.write(data[i])
                f.close()
    # print(len(data))


if __name__ == '__main__':
    # Generate data
    # init_data()
    # Divide the samples
    # divide_data()
    # Shuffle data
    shuffle_data()
