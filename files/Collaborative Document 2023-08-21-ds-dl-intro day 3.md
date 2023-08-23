![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-08-21-ds-dl-intro day 3

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2023-08-21-ddi-day3)

Collaborative Document day 1: [link](https://tinyurl.com/2023-08-21-ddi-day1)

Collaborative Document day 2: [link](https://tinyurl.com/2023-08-21-ddi-day2)

Collaborative Document day 3: [link](https://tinyurl.com/2023-08-21-ddi-day3)

Collaborative Document day 4: [link](https://tinyurl.com/2023-08-21-ddi-day4)  

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

https://esciencecenter-digital-skills.github.io/2023-08-21-ds-dl-intro/

🛠 Setup

https://esciencecenter-digital-skills.github.io/2023-08-21-ds-dl-intro/#setup

## 👩‍🏫👩‍💻🎓 Instructors

Pranav Chandramouli, Sven van der Burg, Laura Ootes

## 🧑‍🙋 Helpers

Laura Ootes, Luisa Orozco

## :ice_cream: :icecream: Icebreaker

Sven's special!

## 🗓️ Agenda
|  Time |                        Topic |     |
| -----:| ----------------------------:| --- |
|  9:00 |       Welcome and icebreaker |     |
|  9:15 | Monitor the training process |     |
| 10:15 |                 Coffee break |     |
| 10:30 | Monitor the training process | :on:|
| 11:30 |                 Coffee break |     |
| 11:45 |          Advance Layer Types |     |
| 12:45 |                      Wrap-up |     |
| 13:00 |                          END |     |

## 🧠 Collaborative Notes & 🔧 Exercises

### 🔧 Exercise 1 Try to reduce the degree of overfitting by lowering the number of parameters
We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

* Is it possible to get rid of overfitting this way?
* Does the overall performance suffer or does it mostly stay the same?
* How low can you go with the number of parameters without notable effect on the performance on the validation set?

> **Solution**
> Adapt the function `create_nn` to take some input parameters `nodes1` and `nodes2`.
```python
def create_nn(nodes1=100, nodes2=50):
   # Input layer
   inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')
   # Dense layers
   layers_dense = keras.layers.Dense(nodes1, 'relu')(inputs)
   layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)
   # Output layer
   outputs = keras.layers.Dense(1)(layers_dense)
   return keras.Model(inputs=inputs, outputs=outputs, name="model_small")
```
>Then call the function with `nodes1=10` and `nodes2=5`
```python
model = create_nn(10, 5)
model.summary()
```

#### Early stopping in keras 
- Very good to avoid over-fitting.

Let's create a new model:
```python
model = create_nn()
compile_model(model)

from tensorflow.keras.callbacks import EarlyStopping
earlystopper = EarlyStopping(
    monitor='val_loss',
    patience=10
    )
```
- `monitor='val_loss'`: metric that is going to be considered.
- `patience=10`: We wait for 10 epochs to see if `val_loss` will no decrease further.

Now we train the model again:
```python
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])
```
:bulb: Note that:
- We still set the number of epochs, but it may stop before reaching the end.
- There is a new parameter `callbacks` in which we pass the `earlystopper` that we have created.

Now we plot the history:
```python
plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

#### Batchnorm
- Normalization inside the network.
- This is different from the data normalization that you may do as a pre-processing step.

```python
# Option 1:
from tensorflow.keras.layers import BatchNormalization

def create_nn():
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    # Option 1:
    batchNorm = BatchNormalization()(inputs)
    # Option 2:
    batchNorm = keras.layers.BatchNormalization()(inputs)
    layers_dense_1 = keras.layers.Dense(100, 'relu')(batchNorm)
    layers_dense_2 = keras.layers.Dense(50, 'relu')(layers_dense_1)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense_2)

    # Defining the model and compiling it
    return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")

model = create_nn()
compile_model(model)
model.summary()
```

:eyes: We see now some Non-trainable params: 89 nodes in the `batch_normalization` layer and there are 2 parameters per node. Total: 178 Non-trainable params

```python
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 1000,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper])

plot_history(history, ['root_mean_squared_error', 'val_root_mean_squared_error'])
```

When you are ready to save your model:
```python
model.save('weather_prediction_v1.0')
```

#### Tensorboard Demo :hotsprings:

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # You can adjust this to add a more meaningful model name
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(X_train, y_train,
                   batch_size = 32,
                   epochs = 200,
                   validation_data=(X_val, y_val),
                   callbacks=[tensorboard_callback],
                   verbose = 2)
```

:rocket: Launch tensorboard:
```python
%load_ext tensorboard
%tensorboard --logdir logs/fit
```
You will see an interface like this:
![tensorboard](https://carpentries-incubator.github.io/deep-learning-intro/fig/03_tensorboard.png)

### 🔧 Exercise 2: Simplify the model and add data + next steps
You may have been wondering why we are including weather observations from multiple cities to predict sunshine hours only in Basel. The weather is a complex phenomenon with correlations over large distances and time scales, but what happens if we limit ourselves to only one city?

1. Since we will be reducing the number of features quite significantly,
we should afford to include more data. Instead of using only 3 years, use 8 or 9 years!
2. Remove all cities from the training data that are not for Basel.
You can use something like:
```python
cols = [c for c in X_data.columns if c[:5] == 'BASEL']
X_data = X_data[cols]
```
3. Now rerun the last model we defined which included the BatchNorm layer.
Recreate the scatter plot comparing your prediction with the baseline prediction based on yesterday's sunshine hours, and compute also the RMSE.
Note that even though we will use many more observations than previously, the network should still train quickly because we reduce the number of features (columns).
Is the prediction better compared to what we had before?

**What could be next steps to further improve the model?**

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results. Usually models are "well behaving" in the sense that small changes to the architectures also only result in small changes of the performance (if any). It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
Applying common sense is often a good first step to make a guess of how much better results *could* be. In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision. But how much better our model could be exactly, often remains difficult to answer.

4. What changes to the model architecture might make sense to explore?
5. Ignoring changes to the model architecture, what might notably improve the prediction quality?
6. (Optional) Try to train a model on all years that are available, and all features from all cities. How does it perform?
7. (Optional) Try one of the most fruitful ideas you have. Does it improve the model?

#### Keypoints
- Changing a hyperparameter (such as the number of nodes) and checking whether it improves performance is called **hyperparameter tuning**.
- Use your validation dat to refine the model and do hyperparameter tuning.
- Use the test data only once in a while to evaluate your model.

### Advaced layer types

#### Types of layers:
- Convolution
- Pooling
- Flatten
- Dropout $\to$ Used to regularize the model

We are going to follow our Deep Learning workflow.
#### 1. Outline the problem
Image classification using the CIFAR10 dataset.

```python
# Load CIFAR-10 dataset
from tensorflow import keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
```

If you get error `[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1125)`
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

We'll take a subset of the dataset:
```python
n = 5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

### 🔧 Exercise 3: Number of features CIFAR-10
How many features does one image in the CIFAR-10 dataset have?

* A. 32
* B. 1024
* C. 3072 :heavy_check_mark: There are 1024 pixels in one image (32 * 32), each pixel has 3 channels (RGB). So 1024 * 3 = 3072
* D. 5000

#### 2. Identify inputs and outputs
Let's explore the data
```python
train_images.shape # check the size of the dataset
> (5000, 32, 32, 3)
train_images.min(), train_images.max()
> (0, 255)
train_labels.shape
> (5000, 1)
train_labels.min(), train_labels.max()
> (0, 9) # 10 classes
```

#### 3. Prepare the data
Scale the data [0, 1]
```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

#### 4. Choose a pretrained model or start building architecture from scratch

### 🔧 Exercise 4: Number of parameters
Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?

* A. 307200
* B. 307300 :heavy_check_mark: 
* C. 100
* D. 3072

> **Solution**
> B: Each entry of the input dimensions, i.e. the shape of one single data point, is connected with 100 neurons of our hidden layer, and each of these neurons has a bias term associated to it. So we have 307300 parameters to learn.

**Convolution**
![Convolutions](https://carpentries-incubator.github.io/deep-learning-intro/fig/04_conv_matrix.png)

![](https://upload.wikimedia.org/wikipedia/commons/1/19/2D_Convolution_Animation.gif?20130203224852)

### 🔧 Exercise 5: Number of model parameters
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise

> **Solution**
> We have 100 matrices with 3 * 3 * 3 = 27 values each so that gives 27 * 100 = 2700 weights. This is a magnitude of 100 less than the fully connected layer with 100 units!

## 📚 Resources
#### Monitoring training process
- [What is a baseline?](https://machinelearningmastery.com/how-to-get-baseline-results-and-why-they-matter/)
- [Popular activation functions](https://iq.opengenus.org/activation-functions-ml/)
- [Keras Batch Normalization Layer](https://keras.io/api/layers/normalization_layers/batch_normalization/)
- [Weights and Biases](gttps://www.wandb.ai)
- [Tensorboard](https://www.tensorflow.org/tensorboard)
- [Hyperparameter search - keras tuner](https://keras.io/keras_tuner/)

#### Advanced layer types
- [Image Kernels Explained](https://setosa.io/ev/image-kernels/)
- [Convolution Neural Network Cheat Sheet](https://iq.opengenus.org/convolution-filters/)
- [Pooling layers](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
- [CIFAR10 paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)


#### Explainable AI
- Example of explainable AI software: [DIANNA](https://github.com/dianna-ai/dianna)

#### Transfer learning & online learning
- [Transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
- [online learning](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5)

#### Different number of features in train and test set
* Inherintly this does not work, you have to pass the exact same number of features to your input layer.
* You can circumvent this by mapping the dimensions of your test data somehow to the dimensions of your training data. The simplest way would be to impute some of the features missing in your test data based on the values in the train data. But you could even train a neural network to do this mapping!