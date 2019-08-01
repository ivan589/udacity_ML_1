
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

    WARNING: Logging before flag parsing goes to stderr.
    W0801 10:08:48.636358 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0801 10:08:48.647337 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0801 10:08:48.650693 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:4185: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.
    
    W0801 10:08:48.677102 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:174: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.
    
    W0801 10:08:48.677669 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:181: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    W0801 10:08:48.698510 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:1834: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    W0801 10:08:48.755431 4552115648 deprecation_wrapper.py:119] From /Users/ivan/anaconda3/lib/python3.7/site-packages/keras/backend/tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    


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

    100%|██████████| 6680/6680 [00:40<00:00, 165.90it/s]
    100%|██████████| 835/835 [00:06<00:00, 137.61it/s]
    100%|██████████| 836/836 [00:05<00:00, 140.21it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.

model = Sequential()

# STEP 1 - Convolution: 
# Extract features from the input image. 
# Convolution preserves the spatial relationship between pixels by learning image features 
# using small squares of input data
model.add( Conv2D(32, (2, 2), input_shape=train_tensors.shape[1:], activation='relu') )

# STEP 2 - Pooling or Downsampling:
# Reduces the dimensionality of each feature map but retains the most important information
model.add( MaxPooling2D(pool_size=(2, 2)) )

model.add( Conv2D(64, (2, 2), input_shape=train_tensors.shape[1:], activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )

# STEP 3 - Flattening:
# The matrix is converted into a linear array (INPUT of the nodes of the neural network)
model.add( GlobalAveragePooling2D() )


# STEP 4 - Connection:
# Connect the Convolutional Network into the Neural Network

# Last layer of the Neural Network has as many nodes as dog categories
model.add(Dense(len(dog_names), activation='softmax'))


model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 223, 223, 32)      416       
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 110, 110, 64)      8256      
    _________________________________________________________________
    max_pooling2d_15 (MaxPooling (None, 55, 55, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_9 ( (None, 64)                0         
    _________________________________________________________________
    dense_14 (Dense)             (None, 133)               8645      
    =================================================================
    Total params: 17,317
    Trainable params: 17,317
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

    W0801 10:14:14.182360 4552115648 deprecation.py:323] From /Users/ivan/.local/lib/python3.7/site-packages/tensorflow/python/ops/math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where


    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6680/6680 [==============================] - 102s 15ms/step - loss: 4.8838 - acc: 0.0093 - val_loss: 4.8677 - val_acc: 0.0108
    
    Epoch 00001: val_loss improved from inf to 4.86773, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 2/5
    6680/6680 [==============================] - 105s 16ms/step - loss: 4.8645 - acc: 0.0130 - val_loss: 4.8539 - val_acc: 0.0168
    
    Epoch 00002: val_loss improved from 4.86773 to 4.85386, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 3/5
    6680/6680 [==============================] - 105s 16ms/step - loss: 4.8386 - acc: 0.0160 - val_loss: 4.8320 - val_acc: 0.0192
    
    Epoch 00003: val_loss improved from 4.85386 to 4.83200, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 4/5
    6680/6680 [==============================] - 107s 16ms/step - loss: 4.8082 - acc: 0.0190 - val_loss: 4.8138 - val_acc: 0.0228
    
    Epoch 00004: val_loss improved from 4.83200 to 4.81383, saving model to saved_models/weights.best.from_scratch.hdf5
    Epoch 5/5
    6680/6680 [==============================] - 107s 16ms/step - loss: 4.7820 - acc: 0.0210 - val_loss: 4.7958 - val_acc: 0.0156
    
    Epoch 00005: val_loss improved from 4.81383 to 4.79577, saving model to saved_models/weights.best.from_scratch.hdf5





    <keras.callbacks.History at 0x1a452cd320>



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

    Test accuracy: 2.0335%


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
    global_average_pooling2d_10  (None, 512)               0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 133)               68229     
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
    6680/6680 [==============================] - 1s 181us/step - loss: 12.1376 - acc: 0.1317 - val_loss: 10.3619 - val_acc: 0.2419
    
    Epoch 00001: val_loss improved from inf to 10.36192, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 2/20
    6680/6680 [==============================] - 1s 83us/step - loss: 9.6520 - acc: 0.3030 - val_loss: 9.5252 - val_acc: 0.3042
    
    Epoch 00002: val_loss improved from 10.36192 to 9.52519, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 3/20
    6680/6680 [==============================] - 1s 83us/step - loss: 8.9367 - acc: 0.3769 - val_loss: 9.2966 - val_acc: 0.3281
    
    Epoch 00003: val_loss improved from 9.52519 to 9.29658, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 4/20
    6680/6680 [==============================] - 1s 82us/step - loss: 8.6131 - acc: 0.4115 - val_loss: 9.0552 - val_acc: 0.3461
    
    Epoch 00004: val_loss improved from 9.29658 to 9.05518, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 5/20
    6680/6680 [==============================] - 1s 85us/step - loss: 8.2761 - acc: 0.4359 - val_loss: 8.7261 - val_acc: 0.3737
    
    Epoch 00005: val_loss improved from 9.05518 to 8.72608, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 6/20
    6680/6680 [==============================] - 1s 85us/step - loss: 8.0030 - acc: 0.4632 - val_loss: 8.6060 - val_acc: 0.3868
    
    Epoch 00006: val_loss improved from 8.72608 to 8.60595, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 7/20
    6680/6680 [==============================] - 1s 93us/step - loss: 7.8060 - acc: 0.4846 - val_loss: 8.5283 - val_acc: 0.3844
    
    Epoch 00007: val_loss improved from 8.60595 to 8.52833, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 8/20
    6680/6680 [==============================] - 1s 86us/step - loss: 7.5708 - acc: 0.4988 - val_loss: 8.1286 - val_acc: 0.4144
    
    Epoch 00008: val_loss improved from 8.52833 to 8.12863, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 9/20
    6680/6680 [==============================] - 1s 90us/step - loss: 7.3586 - acc: 0.5192 - val_loss: 8.2164 - val_acc: 0.4168
    
    Epoch 00009: val_loss did not improve from 8.12863
    Epoch 10/20
    6680/6680 [==============================] - 1s 87us/step - loss: 7.2732 - acc: 0.5293 - val_loss: 8.0208 - val_acc: 0.4180
    
    Epoch 00010: val_loss improved from 8.12863 to 8.02079, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 11/20
    6680/6680 [==============================] - 1s 86us/step - loss: 7.0945 - acc: 0.5400 - val_loss: 7.9635 - val_acc: 0.4204
    
    Epoch 00011: val_loss improved from 8.02079 to 7.96347, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 12/20
    6680/6680 [==============================] - 1s 90us/step - loss: 6.8003 - acc: 0.5464 - val_loss: 7.5285 - val_acc: 0.4503
    
    Epoch 00012: val_loss improved from 7.96347 to 7.52850, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 13/20
    6680/6680 [==============================] - 1s 87us/step - loss: 6.5498 - acc: 0.5713 - val_loss: 7.3943 - val_acc: 0.4647
    
    Epoch 00013: val_loss improved from 7.52850 to 7.39433, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 14/20
    6680/6680 [==============================] - 1s 85us/step - loss: 6.3973 - acc: 0.5852 - val_loss: 7.3724 - val_acc: 0.4695
    
    Epoch 00014: val_loss improved from 7.39433 to 7.37241, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 15/20
    6680/6680 [==============================] - 1s 86us/step - loss: 6.2456 - acc: 0.5931 - val_loss: 7.0533 - val_acc: 0.4802
    
    Epoch 00015: val_loss improved from 7.37241 to 7.05328, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 16/20
    6680/6680 [==============================] - 1s 93us/step - loss: 6.0167 - acc: 0.6060 - val_loss: 7.0415 - val_acc: 0.4778
    
    Epoch 00016: val_loss improved from 7.05328 to 7.04155, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 17/20
    6680/6680 [==============================] - 1s 86us/step - loss: 5.9314 - acc: 0.6175 - val_loss: 6.9152 - val_acc: 0.4970
    
    Epoch 00017: val_loss improved from 7.04155 to 6.91520, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 18/20
    6680/6680 [==============================] - 1s 86us/step - loss: 5.8804 - acc: 0.6219 - val_loss: 6.9135 - val_acc: 0.5018
    
    Epoch 00018: val_loss improved from 6.91520 to 6.91346, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 19/20
    6680/6680 [==============================] - 1s 92us/step - loss: 5.8216 - acc: 0.6232 - val_loss: 6.8787 - val_acc: 0.4802
    
    Epoch 00019: val_loss improved from 6.91346 to 6.87866, saving model to saved_models/weights.best.VGG16.hdf5
    Epoch 20/20
    6680/6680 [==============================] - 1s 89us/step - loss: 5.6179 - acc: 0.6349 - val_loss: 6.8219 - val_acc: 0.5042
    
    Epoch 00020: val_loss improved from 6.87866 to 6.82195, saving model to saved_models/weights.best.VGG16.hdf5





    <keras.callbacks.History at 0x1a41a01ac8>



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

    Test accuracy: 48.0861%


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

I used the `Xception` model, which gives nice results. The CNN used is similar to before with one less Dense layer and slightly higher Dropout rate to avoid overfitting, but gaining performance from one less layer. Due to the bigger amount of Total Params, this is a good candidate to use transfer learning to speed up te tranining time.




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
    global_average_pooling2d_11  (None, 2048)              0         
    _________________________________________________________________
    dense_16 (Dense)             (None, 1024)              2098176   
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_17 (Dense)             (None, 133)               136325    
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
    6680/6680 [==============================] - 7s 998us/step - loss: 0.4187 - acc: 0.8861 - val_loss: 0.8088 - val_acc: 0.8228
    
    Epoch 00001: val_loss improved from inf to 0.80878, saving model to saved_models/weights.best.Xception.hdf5
    Epoch 2/5
    6680/6680 [==============================] - 6s 874us/step - loss: 0.3431 - acc: 0.9042 - val_loss: 0.7878 - val_acc: 0.8371
    
    Epoch 00002: val_loss improved from 0.80878 to 0.78777, saving model to saved_models/weights.best.Xception.hdf5
    Epoch 3/5
    6680/6680 [==============================] - 5s 776us/step - loss: 0.3090 - acc: 0.9189 - val_loss: 0.8694 - val_acc: 0.8275
    
    Epoch 00003: val_loss did not improve from 0.78777
    Epoch 4/5
    6680/6680 [==============================] - 5s 798us/step - loss: 0.2838 - acc: 0.9246 - val_loss: 0.8649 - val_acc: 0.8311
    
    Epoch 00004: val_loss did not improve from 0.78777
    Epoch 5/5
    6680/6680 [==============================] - 6s 907us/step - loss: 0.2624 - acc: 0.9298 - val_loss: 0.9795 - val_acc: 0.8395
    
    Epoch 00005: val_loss did not improve from 0.78777





    <keras.callbacks.History at 0x1a35b3b2b0>



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

    Test acc: 83.13397129186603


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
     'images/Curly-coated_retriever_03896.jpg'
     'images/istockphoto-910856488-612x612.jpg'
     'images/istockphoto-183758982-612x612.jpg' 'images/Brittany_02625.jpg'
     'images/Labrador_retriever_06457.jpg'
     'images/Labrador_retriever_06455.jpg' 'images/sample_cnn.png'
     'images/Welsh_springer_spaniel_08203.jpg']



![png](output_65_1.png)


    I found a HUMAN of breed Australian_shepherd.



![png](output_65_3.png)


    ERROR: No human or dog found!



![png](output_65_5.png)


    I found a DOG of breed Labrador_retriever.



![png](output_65_7.png)


    I found a DOG of breed American_water_spaniel.



![png](output_65_9.png)


    I found a DOG of breed Curly-coated_retriever.



![png](output_65_11.png)


    I found a HUMAN of breed Dachshund.



![png](output_65_13.png)


    ERROR: No human or dog found!



![png](output_65_15.png)


    I found a DOG of breed Brittany.



![png](output_65_17.png)


    I found a DOG of breed Labrador_retriever.



![png](output_65_19.png)


    I found a DOG of breed Chesapeake_bay_retriever.



![png](output_65_21.png)


    ERROR: No human or dog found!



![png](output_65_23.png)


    I found a DOG of breed Welsh_springer_spaniel.



```python
"The results are failry OK. It seems that it has more trouble for photos that the dog or human head is not front facing "
```




    'The results are failry OK. It seems that it has more trouble for photos that the dog or human head is not front facing '




```python

```
