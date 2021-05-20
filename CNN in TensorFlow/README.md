## Convolutional Neural Networks in TensorFlow

# dataset
- Dogs and Cats: https://www.kaggle.com/c/dogs-vs-cats
- Format of data:
    - train_dir
        - cat
        - dog 
    - test_dir
        - cat
        - dog 

# install requirements libraries.
```
pip3 install -r requirements.txt
```
# data augmentation
- Image augmentation and data augmentation is one of the most widely used tools in deep learning to increase dataset size and make neural networks perform better
- Augmentation simply amends images on-the-fly while training using transforms like rotation, scale, flip, ...
- After use Image augmentation, increase accuracy from 77% to 83%. Visualize accuracy and loss per epoch
# using dropouts
- Remove a random number of neurons in your neural network. Using dropouts to make network more efficient in preventing over-specialization and this overfitting
# training model
- custom model:
    ```
    python3 train.py 
    ```
- transfer learning: using transfer learning from a pre-trained network. A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task. I used mobilenet for retrain
    ```
    python3 mobilenet_train.py 
    ```
# inference
- Predict image of cat or dog with:
    ```
    python3 predict.py 
    ```
- File weight of model:
    https://drive.google.com/file/d/1lM8pSnaIHaSDa3Dp0oZE7LouoHsErDSf/view?usp=sharing

- File weight of pretrain mobilenet after training:
    https://drive.google.com/file/d/1-amJCa1S_r-_vGJ8CsXUNQahcAUQuNLN/view?usp=sharing