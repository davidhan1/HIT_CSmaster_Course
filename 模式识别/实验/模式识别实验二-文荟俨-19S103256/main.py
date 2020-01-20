import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D


def mean_square(output, y):
    return np.sum((y - output) ** 2) / 2.0


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


# ReLu 函数
def ReLuFunc(x):
    x = (np.abs(x) + x) / 2.0
    return x


def load_data(train_path):
    with open(train_path, "r") as f:
        data = f.readlines()
    f.close()
    sample = np.zeros((len(data), 3))
    label = np.zeros(len(data))
    for i in range(len(data)):
        sample[i] = eval(data[i])[0:3]
        label[i] = eval(data[i])[3]
    return sample, label


def bn(sample, label):
    # Parameters
    train_epoch = 60
    learning_rate = 0.01

    # Init the network
    w1 = np.random.rand(3, 3)
    bias1 = np.random.rand(3)
    w2 = np.random.rand(4, 3)
    bias2 = np.random.rand(4)

    epoch_iterator = []
    loss_iterator = []
    acc_iterator = []
    testacc_iterator = []
    #Training
    for epoch in range(train_epoch):
        # Validation
        acc = 0
        for n in range(len(sample)):
            net1 = np.zeros(3)
            net2 = np.zeros(4)
            x = sample[n]
            y = label[n]
            # Hidden layer
            for j in range(0, 3):
                for i in range(0, 3):
                    net1[j] = net1[j] + w1[j][i] * x[i]
                net1[j] += bias1[j]
            fnet1 = ReLuFunc(net1)
            # fnet1 = net1

            # Output layer
            for k in range(0, 4):
                for j in range(0, 3):
                    net2[k] = net2[k] + w2[k][j] * fnet1[j]
                net2[k] += bias2[k]
            output = softmax(net2)
            if np.argmax(output) == y:
                acc += 1
        # Validation
        test_path = 'test_samples.txt'
        test_sample, test_label = load_data(test_path)
        test_acc = 0
        for n in range(len(test_sample)):
            net1 = np.zeros(3)
            net2 = np.zeros(4)
            x = test_sample[n]
            y = test_label[n]
            # Hidden layer
            for j in range(0, 3):
                for i in range(0, 3):
                    net1[j] = net1[j] + w1[j][i] * x[i]
                net1[j] += bias1[j]
            fnet1 = ReLuFunc(net1)

            # Output layer
            for k in range(0, 4):
                for j in range(0, 3):
                    net2[k] = net2[k] + w2[k][j] * fnet1[j]
                net2[k] += bias2[k]
            output = softmax(net2)
            if np.argmax(output) == y:
                test_acc += 1

        if epoch % 2 == 0:
            learning_rate *= 0.95
        sum_loss = 0
        # rand_num = random.randint(0, len(label))
        # #print(rand_num)
        # random.seed(rand_num)
        # random.shuffle(sample)
        # random.seed(rand_num)
        # random.shuffle(label)
        for n in range(len(sample)):
            x = sample[n]
            #print(x)
            y = label[n]
            # Convert to one_hot
            if y == 0:
                y = np.array([1, 0, 0, 0], dtype=np.float)
            elif y == 1:
                y = np.array([0, 1, 0, 0], dtype=np.float)
            elif y == 2:
                y = np.array([0, 0, 1, 0], dtype=np.float)
            elif y == 3:
                y = np.array([0, 0, 0, 1], dtype=np.float)
            # Forword
            net1 = np.zeros(3)
            net2 = np.zeros(4)

            # Hidden layer
            for j in range(0, 3):
                for i in range(0, 3):
                    net1[j] = net1[j] + w1[j][i] * x[i]
                net1[j] += bias1[j]
            fnet1 = ReLuFunc(net1)


            # Output layer
            for k in range(0, 4):
                for j in range(0, 3):
                    net2[k] = net2[k] + w2[k][j] * fnet1[j]
                net2[k] += bias2[k]
            output = softmax(net2)

            # Define loss
            loss = mean_square(output, y)
            sum_loss += loss

            # Back
            det_k = np.zeros(4)
            for k in range(0, 4):
                det_k[k] = (y[k] - output[k]) * (output[k] * (1 - output[k]))
            det_j = np.zeros(3)
            for j in range(0, 3):
                sum = 0
                for k in range(0, 4):
                    sum = sum + w2[k][j] * det_k[k]
                if net1[j] >= 0:
                    det_j[j] = sum
                else:
                    det_j[j] = 0
            # Update w and b
            for j in range(0, 3):
                for i in range(0, 3):
                    w1[j][i] = w1[j][i] + learning_rate * det_j[j] * x[i]
                bias1[j] = bias1[j] + learning_rate * det_j[j]

            for k in range(0, 4):
                for j in range(0, 3):
                    w2[k][j] = w2[k][j] + learning_rate * det_k[k] * fnet1[j]
                bias2[k] = bias2[k] + learning_rate * det_k[k]

        epoch_iterator.append(epoch)
        loss_iterator.append(sum_loss/len(label))
        acc_iterator.append(acc*1.0/len(label))
        testacc_iterator.append(test_acc*1.0/len(test_label))
        print("epoch = {0}, loss = {1}, train_acc = {2}, test_acc = {3}".format(epoch, sum_loss/len(label), acc*1.0/len(label), test_acc*1.0/len(test_label)))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # 做镜像处理
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
    ax1.set_xlabel('训练轮数', fontproperties=font)  # 设置x轴标题
    ax1.set_ylabel('loss值', fontproperties=font)  # 设置Y1轴标题
    ax2.set_ylabel('准确率', fontproperties=font)  # 设置Y2轴标题
    ax3 = ax2.twiny()
    p1, = ax1.plot(epoch_iterator, loss_iterator, '-')
    p2, = ax2.plot(epoch_iterator, acc_iterator, '--')
    p3, = ax3.plot(epoch_iterator, testacc_iterator, ':')
    plt.legend(handles=[p1, p2, p3], labels=['训练loss', '训练集准确率', '测试集准确率'], loc=5, prop=font)
    plt.draw()
    plt.show()
    return w1, w2, bias1, bias2

def inference(w1, w2, b1, b2, test_sample):
    for n in range(0, len(test_sample)):
        net1 = np.zeros(3)
        net2 = np.zeros(4)
        x = test_sample[n]
        # Hidden layer
        for j in range(0, 3):
            for i in range(0, 3):
                net1[j] = net1[j] + w1[j][i] * x[i]
            net1[j] += b1[j]
        fnet1 = ReLuFunc(net1)

        # Output layer
        for k in range(0, 4):
            for j in range(0, 3):
                net2[k] = net2[k] + w2[k][j] * fnet1[j]
            net2[k] += b2[k]
        output = softmax(net2)
        index = np.argmax(output)
        print("Sample {0}, inference label:{1}, and its probability:{2}".format(n, index, output[index]))


def cal_p(x):
    u = np.asarray([[0, 0, 0],
                   [0, 1, 0],
                   [-1, 0, 1],
                   [0, 0.5, 1]])
    sigma = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       [[1, 0, 1], [0, 2, 2], [1, 2, 5]],
                       [[2, 0, 0], [0, 6, 0], [0, 0, 1]],
                       [[2, 0, 0], [0, 1, 0], [0, 0, 3]]])
    p = np.zeros(4)
    for i in range(0, 4):
        det = np.linalg.det(sigma[i])
        inv = np.linalg.inv(sigma[i])
        xu_ = np.transpose(x - u[i])
        xu = np.transpose(xu_)
        first = np.dot(xu_, inv)
        second = np.dot(first, xu)
        d = len(x)
        p[i] = 1.0/((2*np.pi)**(d/2.0)*np.sqrt(det))*np.exp(-second/2.0)
    prob = max(p)/np.sum(p)
    return prob, np.argmax(p)


if __name__ == '__main__':
    train_path = 'samples_shuffle.txt'
    sample, label = load_data(train_path)
    w1, w2, b1, b2 = bn(sample, label)
    test_sample = np.array([[0, 0, 0],
                            [-1, 0, 1],
                            [0.5, -0.5, 0],
                            [-1, 0, 0],
                            [0, 0, -1]])
    inference(w1, w2, b1, b2, test_sample)


    # Calculate the probability by using Gauss function
    '''
    for n in range(0, len(test_sample)):
        p, index = cal_p(test_sample[n])
        print("Sample {0}, inference label:{1}, and its probability:{2}".format(n, index, p))
    '''

    # Plot the initial samples
    '''
    sample_path = 'samples.txt'
    sample, label = load_data(sample_path)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(sample[:1100, 0], sample[:1100, 1], sample[:1100, 2], c='black', s=5)  # 绘制数据点
    ax.scatter(sample[1100:2200, 0], sample[1100:2200, 1], sample[1100:2200, 2], c='r', s=5)  # 绘制数据点
    ax.scatter(sample[2200:3300, 0], sample[2200:3300, 1], sample[2200:3300, 2], c='g', s=5)  # 绘制数据点
    ax.scatter(sample[3300:4400, 0], sample[3300:4400, 1], sample[3300:4400, 2], c='y', s=5)  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    '''