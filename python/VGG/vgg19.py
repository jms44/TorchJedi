def computeMeanPixelFrom(images):
    sumPixel = np.array([0,0,0])
    means = np.zeros((len(images), 3))
    for i in range(len(images)):
        img = cv2.imread(images[i])
        mean = np.mean(img, axis=(0,1))
        means[i] = mean
    return np.mean(means, axis=0)/255.0

class vgg19:
    def __init__():
        pass
