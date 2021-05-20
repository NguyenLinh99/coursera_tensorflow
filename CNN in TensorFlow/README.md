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
# training model
```
python3 train.py 
```
# inference
```
python3 predict.py 
```
# data augmentation
- Image augmentation and data augmentation is one of the most widely used tools in deep learning to increase dataset size and make neural networks perform better
- Augmentation simply amends images on-the-fly while training using transforms like rotation, scale, flip, ...
- After use Image augmentation, increase accuracy from 77% to 83%. Visualize accuracy and loss per epoch

