import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from matplotlib.colors import ListedColormap

def showData(traindata, colors):
    classColormap = ListedColormap(colors)
    pl.scatter([traindata[i][0][0] for i in range(len(traindata))],
               [traindata[i][0][1] for i in range(len(traindata))],
               c=[traindata[i][1] for i in range(len(traindata))],
               cmap=classColormap)

def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def calculate_acc(data1, data2):
    return sum([int(data1[i] == data2[i]) for i in range(len(data1))]) / float(len(data1))

def findNearestPoints(point, data):
    nearestpoint = []
    for i in data:
        d = dist(point, eval(str(i[0])))  # 计算测试样本和训练样本的欧氏距离
        nearestpoint.append([d, i[1]])
    return sorted(nearestpoint)

def knn(testData, trainXY, k):
    textXY_inference = []
    for i in range(0, len(testData)):
        nearestpoints = findNearestPoints(testData[i], trainXY)
        temp = []
        for j in range(0, k):
            temp.append(nearestpoints[j][1])
        dict = {}
        for key in temp:
            dict[key] = dict.get(key, 0) + 1
        max_prices = max(zip(dict.values(), dict.keys()))
        textXY_inference.append(max_prices[1])
    return textXY_inference

def inference(trainXY, testX, testlabel, k):
    testWithLabels = []
    testData = np.array(testX)
    textXY_inference = knn(testData, trainXY, k)
    for i in range(len(testX)):
        testWithLabels.append([testX[i], testlabel[i]])
    a = calculate_acc(textXY_inference, testlabel)
    print("acc = ", a)
    return testWithLabels

if __name__ == '__main__':
    # parameters
    category = 3
    num_samples = 500
    num_train = 1000
    k = 800
    num_test = 500
    new_data = []
    with open("./samples2.txt", "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        new_data.append(eval(data[i]))
    trainXY = new_data[0:num_train]
    testXY = new_data[1000:1000+num_test]
    testlabel = []
    for i in range(len(testXY)):
        testlabel.append(testXY[i][1])
    testX = [[testXY[i][0][0], testXY[i][0][1]] for i in range(len(testXY))]
    res1 = inference(trainXY, testX, testlabel, k)
    train_color = ['#CD5555', '#104E8B', '#008B00']
    inference_color = ['#FFC1C1', '#BBFFFF', '#9AFF9A']
    showData(trainXY, train_color)
    showData(res1, inference_color)
    #pl.show()
