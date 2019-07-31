
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13233 total human images.


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
humans = 0
dogs = 0

for img in human_files_short:
    humans = (humans + 1) if face_detector(img) else humans
    
for img in dog_files_short:
    dogs = (dogs + 1) if face_detector(img) else dogs

print(f"Humans: {humans}%")
print(f"Dogs: {dogs}%")
    
```

    Humans: 100%
    Dogs: 12%


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__

__In my opinion, this is not a reasonable expectation to pose on the user. One way we could improve trhe reults is to add more images to our training set that contain humans in poor light conditions to better train the model.__


We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

humans = 0
dogs = 0

for img in human_files_short:
    humans = (humans + 1) if dog_detector(img) else humans
    
for img in dog_files_short:
    dogs = (dogs + 1) if dog_detector(img) else dogs

print(f"Humans: {humans}%")
print(f"Dogs: {dogs}%")
    
```

    Humans: 0%
    Dogs: 100%


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:46<00:00, 143.06it/s]
    100%|██████████| 835/835 [00:06<00:00, 131.97it/s]
    100%|██████████| 836/836 [00:06<00:00, 132.89it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 

- `GlobalAveragePooling2D` is chosen instead of `Flatten`, this will result in a smaller Dense layer afterwards, which is less expensive and may result in less overfitting. Also `Flatten` will perform better (I think) with bigger data sets.
- Dropout was used before using the fully connected layer to reduce overfitting.


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(train_tensors.shape[1:])))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(dog_names), activation='softmax'))


model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_11  (None, 3)                 0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 1024)              4096      
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_21 (Dense)             (None, 512)               524800    
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 133)               68229     
    =================================================================
    Total params: 597,125
    Trainable params: 597,125
    Non-trainable params: 0
    _________________________________________________________________


### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 5

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6680/6680 [==============================] - 6s 919us/step - loss: 4.8826 - acc: 0.0094 - val_loss: 4.8674 - val_acc: 0.0096
    
    Epoch 00001: val_loss improved from inf to 4.86741, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 2/5
    6680/6680 [==============================] - 5s 726us/step - loss: 4.8648 - acc: 0.0120 - val_loss: 4.8561 - val_acc: 0.0156
    
    Epoch 00002: val_loss improved from 4.86741 to 4.85611, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 3/5
    6680/6680 [==============================] - 5s 728us/step - loss: 4.8381 - acc: 0.0169 - val_loss: 4.8348 - val_acc: 0.0180
    
    Epoch 00003: val_loss improved from 4.85611 to 4.83479, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 4/5
    6680/6680 [==============================] - 5s 735us/step - loss: 4.8214 - acc: 0.0171 - val_loss: 4.8262 - val_acc: 0.0168
    
    Epoch 00004: val_loss improved from 4.83479 to 4.82615, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 5/5
    6680/6680 [==============================] - 5s 738us/step - loss: 4.8106 - acc: 0.0175 - val_loss: 4.8693 - val_acc: 0.0120
    
    Epoch 00005: val_loss did not improve from 4.82615





    <keras.callbacks.History at 0x1a417f1b00>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 1.7943%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_12  (None, 512)               0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6680/6680 [==============================] - 2s 315us/step - loss: 11.8928 - acc: 0.1326 - val_loss: 9.8560 - val_acc: 0.2695
    
    Epoch 00001: val_loss improved from inf to 9.85602, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 2/20
    6680/6680 [==============================] - 1s 132us/step - loss: 9.4739 - acc: 0.3081 - val_loss: 9.1976 - val_acc: 0.3269
    
    Epoch 00002: val_loss improved from 9.85602 to 9.19756, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 3/20
    6680/6680 [==============================] - 1s 132us/step - loss: 8.6657 - acc: 0.3840 - val_loss: 8.6736 - val_acc: 0.3737
    
    Epoch 00003: val_loss improved from 9.19756 to 8.67361, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 4/20
    6680/6680 [==============================] - 1s 145us/step - loss: 8.2767 - acc: 0.4305 - val_loss: 8.5401 - val_acc: 0.3904
    
    Epoch 00004: val_loss improved from 8.67361 to 8.54009, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 5/20
    6680/6680 [==============================] - 1s 136us/step - loss: 8.0922 - acc: 0.4563 - val_loss: 8.3145 - val_acc: 0.4072
    
    Epoch 00005: val_loss improved from 8.54009 to 8.31445, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 6/20
    6680/6680 [==============================] - 1s 131us/step - loss: 7.9098 - acc: 0.4795 - val_loss: 8.3790 - val_acc: 0.4060
    
    Epoch 00006: val_loss did not improve from 8.31445
    Epoch 7/20
    6680/6680 [==============================] - 1s 133us/step - loss: 7.7753 - acc: 0.4909 - val_loss: 8.2909 - val_acc: 0.4144
    
    Epoch 00007: val_loss improved from 8.31445 to 8.29085, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 8/20
    6680/6680 [==============================] - 1s 142us/step - loss: 7.6755 - acc: 0.5013 - val_loss: 8.1201 - val_acc: 0.4287
    
    Epoch 00008: val_loss improved from 8.29085 to 8.12005, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 9/20
    6680/6680 [==============================] - 1s 131us/step - loss: 7.6094 - acc: 0.5096 - val_loss: 8.0812 - val_acc: 0.4455
    
    Epoch 00009: val_loss improved from 8.12005 to 8.08123, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 10/20
    6680/6680 [==============================] - 1s 136us/step - loss: 7.5123 - acc: 0.5142 - val_loss: 7.9118 - val_acc: 0.4299
    
    Epoch 00010: val_loss improved from 8.08123 to 7.91180, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 11/20
    6680/6680 [==============================] - 1s 138us/step - loss: 7.1397 - acc: 0.5280 - val_loss: 7.6474 - val_acc: 0.4623
    
    Epoch 00011: val_loss improved from 7.91180 to 7.64737, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 12/20
    6680/6680 [==============================] - 1s 137us/step - loss: 7.0059 - acc: 0.5509 - val_loss: 7.6966 - val_acc: 0.4587
    
    Epoch 00012: val_loss did not improve from 7.64737
    Epoch 13/20
    6680/6680 [==============================] - 1s 136us/step - loss: 6.9669 - acc: 0.5579 - val_loss: 7.6291 - val_acc: 0.4599
    
    Epoch 00013: val_loss improved from 7.64737 to 7.62912, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 14/20
    6680/6680 [==============================] - 1s 131us/step - loss: 6.8874 - acc: 0.5600 - val_loss: 7.4160 - val_acc: 0.4599
    
    Epoch 00014: val_loss improved from 7.62912 to 7.41604, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 15/20
    6680/6680 [==============================] - 1s 140us/step - loss: 6.7895 - acc: 0.5672 - val_loss: 7.3808 - val_acc: 0.4814
    
    Epoch 00015: val_loss improved from 7.41604 to 7.38081, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 16/20
    6680/6680 [==============================] - 1s 131us/step - loss: 6.7564 - acc: 0.5732 - val_loss: 7.2789 - val_acc: 0.4814
    
    Epoch 00016: val_loss improved from 7.38081 to 7.27889, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 17/20
    6680/6680 [==============================] - 1s 135us/step - loss: 6.7071 - acc: 0.5729 - val_loss: 7.2302 - val_acc: 0.4934
    
    Epoch 00017: val_loss improved from 7.27889 to 7.23019, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 18/20
    6680/6680 [==============================] - 1s 131us/step - loss: 6.6280 - acc: 0.5792 - val_loss: 7.3067 - val_acc: 0.4838
    
    Epoch 00018: val_loss did not improve from 7.23019
    Epoch 19/20
    6680/6680 [==============================] - 1s 132us/step - loss: 6.5147 - acc: 0.5835 - val_loss: 7.2197 - val_acc: 0.4862
    
    Epoch 00019: val_loss improved from 7.23019 to 7.21971, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 20/20
    6680/6680 [==============================] - 1s 143us/step - loss: 6.3634 - acc: 0.5895 - val_loss: 7.1193 - val_acc: 0.4850
    
    Epoch 00020: val_loss improved from 7.21971 to 7.11928, saving model to saved_models/weights.best.VGG16.hdf5





    <keras.callbacks.History at 0x1a44e5b160>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 46.1722%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
train_Xception = bottleneck_features['train']
valid_Xception = bottleneck_features['valid']
test_Xception = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 

I used the `Xception` model, which gives nice results. The CNN used is similar to before with one less Dense layer and slightly higher Dropout rate to avoid overfitting, but gaining performance from one less layer.




```python
### TODO: Define your architecture.

Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=train_Xception.shape[1:]))
Xception_model.add(Dense(1024, activation='relu'))
Xception_model.add(Dropout(0.4))
Xception_model.add(Dense(len(dog_names), activation='softmax'))

Xception_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_20  (None, 2048)              0         
    _________________________________________________________________
    dense_38 (Dense)             (None, 1024)              2098176   
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_39 (Dense)             (None, 133)               136325    
    =================================================================
    Total params: 2,234,501
    Trainable params: 2,234,501
    Non-trainable params: 0
    _________________________________________________________________


### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.
from keras.callbacks import ModelCheckpoint 

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.hdf5', verbose=1, save_best_only=True)

Xception_model.fit(train_Xception, train_targets, 
          validation_data=(valid_Xception, valid_targets),
          epochs=5, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6680/6680 [==============================] - 14s 2ms/step - loss: 1.3595 - acc: 0.6656 - val_loss: 0.7323 - val_acc: 0.7844
    
    Epoch 00001: val_loss improved from inf to 0.73226, saving model to saved_models/weights.best.Xception.hdf5
    Epoch 2/5
    6680/6680 [==============================] - 7s 1ms/step - loss: 0.6822 - acc: 0.8078 - val_loss: 0.7200 - val_acc: 0.8036
    
    Epoch 00002: val_loss improved from 0.73226 to 0.72005, saving model to saved_models/weights.best.Xception.hdf5
    Epoch 3/5
    6680/6680 [==============================] - 7s 1ms/step - loss: 0.5547 - acc: 0.8437 - val_loss: 0.6811 - val_acc: 0.8192
    
    Epoch 00003: val_loss improved from 0.72005 to 0.68115, saving model to saved_models/weights.best.Xception.hdf5
    Epoch 4/5
    6680/6680 [==============================] - 8s 1ms/step - loss: 0.4834 - acc: 0.8626 - val_loss: 0.8405 - val_acc: 0.8060
    
    Epoch 00004: val_loss did not improve from 0.68115
    Epoch 5/5
    6680/6680 [==============================] - 8s 1ms/step - loss: 0.4315 - acc: 0.8787 - val_loss: 0.6993 - val_acc: 0.8347
    
    Epoch 00005: val_loss did not improve from 0.68115





    <keras.callbacks.History at 0x1a965297f0>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.

transfer_predictions = [
    np.argmax(Xception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Xception
]

test_accuracy = 100 * np.sum(np.array(transfer_predictions) == np.argmax(test_targets, axis=1)) / len(transfer_predictions)
print(f'Test acc: {test_accuracy}')
```

    Test acc: 81.9377990430622


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
def predict_dog_breed(img_path):
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    predicted_vector = Xception_model.predict(bottleneck_feature)
    predicted_index = np.argmax(predicted_vector)
    return dog_names[predicted_index]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def human_or_dog(img_path):
    is_dog = dog_detector(img_path)
    is_human = face_detector(img_path)
    dog_breed = predict_dog_breed(img_path)
    
    # plot image with comment
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
    
    if is_dog:
        print(f"I found a DOG of breed {dog_breed}.")
        return dog_breed
    elif is_human:
        print(f"I found a HUMAN of breed {dog_breed}.")
        return dog_breed
    else:
        print("ERROR: No human or dog found!")
        return False
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 

The algorihtm performs much better than I expected. Possible imporovments are: doing hyperparams optimization with a truth matrix, bigger data set, more epochs.


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
import numpy as np
sample_files = np.array(glob("images/*"))
print(sample_files)

for img in sample_files:
    human_or_dog(img)
```

    ['images/sample_human_output.png' 'images/sample_dog_output.png'
     'images/Labrador_retriever_06449.jpg'
     'images/American_water_spaniel_00648.jpg'
     'images/Curly-coated_retriever_03896.jpg' 'images/Brittany_02625.jpg'
     'images/Labrador_retriever_06457.jpg'
     'images/Labrador_retriever_06455.jpg' 'images/sample_cnn.png'
     'images/Welsh_springer_spaniel_08203.jpg']



![png](output_65_1.png)


    I found a HUMAN of breed Havanese.



![png](output_65_3.png)


    ERROR: No human or dog found!



![png](output_65_5.png)


    I found a DOG of breed Labrador_retriever.



![png](output_65_7.png)


    I found a DOG of breed Curly-coated_retriever.



![png](output_65_9.png)


    I found a DOG of breed Curly-coated_retriever.



![png](output_65_11.png)


    I found a DOG of breed Brittany.



![png](output_65_13.png)


    I found a DOG of breed Labrador_retriever.



![png](output_65_15.png)


    I found a DOG of breed Labrador_retriever.



![png](output_65_17.png)


    ERROR: No human or dog found!



![png](output_65_19.png)


    I found a DOG of breed Welsh_springer_spaniel.



```python

```
