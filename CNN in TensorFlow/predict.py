import cv2
import tensorflow as tf
import numpy as np
import glob

## inference
for path in glob.glob("dogs-vs-cats/test1/*.jpg"):
    img = cv2.imread(path) 
    cv2.imshow("img", img)
    img = cv2.resize(img, (64,64)).astype(np.float32)/255
    img = np.expand_dims(img, axis=0)
    # load weight
    model = tf.keras.models.load_model("best_acc.h5")
    pred = model.predict(img)
    print(pred)
    if pred<0.5:
        print("cat")
    else:
        print("dog")
    cv2.waitKey()