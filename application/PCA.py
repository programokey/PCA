import numpy as np
import matplotlib.pyplot as plt

def loadImages(filename="t10k-images.idx3-ubyte"):
    with open(filename, 'rb') as f:
        data = f.read()
        magic = int.from_bytes(data[:4], byteorder= 'big')
        num = int.from_bytes(data[4:8], byteorder= 'big')
        row_size = int.from_bytes(data[8:12], byteorder='big')
        col_size = int.from_bytes(data[12:16], byteorder='big')
        data = data[16:]
        res = np.zeros(num * col_size * row_size)
        for i, pixel in enumerate(data):
            res[i] = pixel/255
        del data
        res = res.reshape((num, -1))
        return res

def PCA(X):
    X -= np.mean(X, 0)
    XTX = np.matmul(np.transpose(X), X)
    V, W = np.linalg.eig(XTX)
    print(W.shape)
    print(V.shape)
    items = [(v, W[i, :]) for i, v in enumerate(V)]
    items.sort(key=lambda x:x[0], reverse=True)
    V = np.array(list(map(lambda x: x[0], items)), dtype=np.float64)
    W = np.array(list(map(lambda x: x[1], items)), dtype=np.float64)
    return V, W

def show_images(images, col_size=28, row_size=28, n=10, pic_name='orgin.png'):
    res = np.zeros((n*row_size, n*col_size))
    for i in range(n):
        for j in range(n):
            for row in range(row_size):
                for col in range(col_size):
                    res[i*row_size + row][j*col_size + col] = \
                        images[i*n + j, row*row_size + col]
    plt.figure()
    plt.imshow(res)
    plt.savefig(pic_name)
    plt.show()

if __name__ == '__main__':
    res = loadImages()
    print(res.shape)
    V, W = PCA(res)
    plt.figure()
    plt.plot(V, 'b')
    # plt.plot(np.log(V + 1), 'r')
    plt.title('eigenvalue of MNIST dataset')
    plt.savefig('eigenvalue.png')
    res = res[:100, :]
    show_images(res, pic_name='orgin.png')
    tmp = np.matmul(res, W[:, :50])
    tmp = np.matmul(tmp, W[:, :50].transpose())
    show_images(tmp, pic_name='reconstruct50.png')
    n = 100
    tmp = np.matmul(res, W[:, :n])
    tmp = np.matmul(tmp, W[:, :n].transpose())
    show_images(tmp, pic_name='reconstruct%d.png' % (n))
    n = 200
    tmp = np.matmul(res, W[:, :n])
    tmp = np.matmul(tmp, W[:, :n].transpose())
    show_images(tmp, pic_name='reconstruct%d.png' % (n))
    n = 300
    tmp = np.matmul(res, W[:, :n])
    tmp = np.matmul(tmp, W[:, :n].transpose())
    show_images(tmp, pic_name='reconstruct%d.png'%(n))

