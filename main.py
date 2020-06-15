from os import path
import glob
import numpy as np
from keras.preprocessing import image

from dcgan import DCGAN

modelSaveLocation = "./saved_model"

def init():
    dcgan = DCGAN()
    if path.exists("%s/generator.h5" % modelSaveLocation):
        print("Model is trained. Loading from %s" % modelSaveLocation)
        dcgan.loadWeights(modelSaveLocation)
        dcgan.generate("./images/generated/example.png")
    else:
        data = loadData()
        print("Model is not trained. Loaded %d images for training" % len(data))
        dcgan.train(data=data, epochs=2000, batch_size=32, save_interval=50)
        print("Model is trained. Saving to %s" % modelSaveLocation)
        dcgan.saveWeights(modelSaveLocation)

def loadData():
    imageFiles = glob.glob("./images/*.png")
    data = []
    for imageFile in imageFiles:
        imageData = image.load_img(imageFile, target_size=(64, 64), color_mode="rgba")
        data.append(image.img_to_array(imageData))
    print(data[0])
    return np.array(data)

if __name__ == "__main__":
    init()
