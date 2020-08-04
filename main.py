import numpy as np
import matplotlib.pyplot as plt
import random
from hopfield import Hopfield

x_0 = np.array([-1, 1, 1, 1, -1, 1, -1, -1, -1, 1,  1, -1, -1, -1, 1,  1, -1, -1, -1, 1, -1, 1, 1, 1, -1])
x_1 = np.array([-1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1])
x_2 = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
x_3 = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
x_4 = np.array([1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1])
x_5 = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
x_list = np.array([x_0, x_1, x_2, x_3, x_4, x_5])


#画像を表示
def show_image(data):
    img = np.reshape(-(data - 1) / 2, (5, 5))
    plt.imshow(img, cmap = 'gray', vmin = 0, vmax = 1, interpolation = 'none')
    plt.show()

#割合を指定して画素を反転させる
def make_noise(data, rate):
    new_data = np.zeros(len(data))
    for i in range(len(data)):  
        rand = random.randint(0,100)
        if rand >= rate:
            new_data[i] = data[i]
        else:
            new_data[i] = data[i] * -1
    return new_data

def train(train_data):
    model = Hopfield(0, 25)
    model.update(train_data)
    return model

def test(model, test_data, noise_rate):
    test_data = make_noise(test_data, noise_rate)
    return model(test_data)

#訓練データの作成
def make_train_data(num_pattern, epoch):
    sample_list = x_list
    train_list = np.empty([epoch, 25])
    for i in range(epoch):
        train_list[i] = sample_list[random.randint(0,num_pattern-1)]
    return train_list

def main():
    train_list = make_train_data(3, 200)
    trained_model = train(train_list)
    result = test(trained_model, x_3, 20)
    show_image(result)

if __name__ == "__main__":
    main()