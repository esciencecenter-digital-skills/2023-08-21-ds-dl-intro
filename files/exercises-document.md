# Episode 1-introduction.Rmd 

:: challenge
# Calculate the output for one neuron
uppose we have:

Input: X = (0, 0.5, 1)
Weights: W = (-1, -0.5, 0.5)
Bias: b = 1
Activation function _relu_: `f(x) = max(x, 0)`

hat is the output of the neuron?

Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._


:: challenge
# Deep Learning Problems Exercise

hich of the following would you apply Deep Learning to?

. Recognising whether or not a picture contains a bird.
. Calculating the median and interquartile range of a dataset.
. Identifying MRI images of a rare disease when only one or two example images available for training.
. Identifying people in pictures after being trained only on cats and dogs.
. Translating English into French.


:: challenge
# Deep Learning workflow exercise

hink about a problem you would like to use Deep Learning to solve.

. What do you want a Deep Learning system to be able to tell you?
. What data inputs and outputs will you have?
. Do you think you will need to train the network or will a pre-trained network be suitable?
. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you will use to train the network.


# Episode 2-keras.Rmd 

:: challenge
# Penguin Dataset

nspect the penguins dataset.

. What are the different features called in the dataframe?
. Are the target classes of the dataset stored as numbers or strings?
. How many samples does this dataset have?


:: challenge
# Pairplot

ake a look at the pairplot we created. Consider the following questions:

Is there any class that is easily distinguishable from the others?
Which combination of attributes shows the best separation for all 3 class labels at once?


:: challenge
# One-hot encoding vs ordinal encoding
. How many output neurons will our network have now that we
ne-hot encoded the target class?
. Another encoding method is 'ordinal encoding'.
ere the variable is represented by a single column,
here each category is represented by a different integer
0, 1, 2 in the case of the 3 penguin species).
ow many output neurons will a network have when ordinal encoding is used?
. (Optional) What would be the advantage of using one-hot versus ordinal encoding
or the task of classifying penguin species?


:: challenge
# Training and Test sets
ake a look at the training and test set we created.
How many samples do the training and test sets have?
Are the classes in the training set well balanced?


:: challenge
# Create the neural network
ith the code snippets above, we defined a Keras model with 1 hidden layer with
0 neurons and an output layer with 3 neurons.

How many parameters does the resulting model have?
What happens to the number of parameters if we increase or decrease the number of neurons
n the hidden layer?


:: challenge
# The Training Curve
ooking at the training curve we have just made.

. How does the training progress?
Does the training loss increase or decrease?
Does it change quickly or slowly?
Does the graph look very jittery?
. Do you think the resulting trained network will work well on the test set?


:: challenge
# Confusion Matrix
easure the performance of the neural network you trained and
isualize a confusion matrix.

Did the neural network perform well on the test set?
Did you expect this from the training loss you saw?
What could we do to improve the performance?


# Episode 3-monitor-the-model.Rmd 

:: challenge
# Exercise: Explore the dataset
et's get a quick idea of the dataset.

How many data points do we have?
How many features does the data have (don't count month and date as a feature)?
What are the different types of measurements (humidity etc.) in the data and how many are there?
(Optional) Plot the amount of sunshine hours in Basel over the course of a year. Are there any interesting properties that you notice?


:: challenge
# Exercise: Architecture of the network
s we want to design a neural network architecture for a regression task,
ee if you can first come up with the answers to the following questions:

. What must be the dimension of our input layer?
. We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the `softmax` activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression?
int: A layer with `relu` activation, with `sigmoid` activation or no activation at all?
. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?


:: challenge
# Exercise: Reflecting on our results
Is the performance of the model as you expected (or better/worse)?
Is there a noteable difference between training set and test set? And if so, any idea why?
(Optional) When developing a model, you will often vary different aspects of your model like
hich features you use, model parameters and architecture. It is important to settle on a
ingle-number evaluation metric to compare your models.
What single-number evaluation metric would you choose here and why?


:: challenge
# Exercise: Baseline
. Looking at this baseline: Would you consider this a simple or a hard problem to solve?
. (Optional) Can you think of other baselines?


:: challenge
# Exercise: plot the training progress.
. Is there a difference between the training curves of training versus validation data? And if so, what would this imply?
. (Optional) Take a pen and paper, draw the perfect training and validation curves.
This may seem trivial, but it will trigger you to think about what you actually would like to see)


:: challenge
# Exercise: Try to reduce the degree of overfitting by lowering the number of parameters
e can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
ry to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
f time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

Is it possible to get rid of overfitting this way?
Does the overall performance suffer or does it mostly stay the same?
How low can you go with the number of parameters without notable effect on the performance on the validation set?


:: challenge
# Exercise: Simplify the model and add data
ou may have been wondering why we are including weather observations from
ultiple cities to predict sunshine hours only in Basel. The weather is
complex phenomenon with correlations over large distances and time scales,
ut what happens if we limit ourselves to only one city?

. Since we will be reducing the number of features quite significantly,
e should afford to include more data. Instead of using only 3 years, use
or 9 years!
. Remove all cities from the training data that are not for Basel.
ou can use something like:
``python
ols = [c for c in X_data.columns if c[:5] == 'BASEL']
_data = X_data[cols]
``
. Now rerun the last model we defined which included the BatchNorm layer.
ecreate the scatter plot comparing your prediction with the baseline
rediction based on yesterday's sunshine hours, and compute also the RMSE.
ote that even though we will use many more observations than previously,
he network should still train quickly because we reduce the number of
eatures (columns).
s the prediction better compared to what we had before?
. (Optional) Try to train a model on all years that are available,
nd all features from all cities. How does it perform?



:: challenge
# Open question: What could be next steps to further improve the model?

ith unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
sually models are "well behaving" in the sense that small changes to the architectures also only result in small changes of the performance (if any).
t is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
pplying common sense is often a good first step to make a guess of how much better results *could* be.
n the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
ut how much better our model could be exactly, often remains difficult to answer.

What changes to the model architecture might make sense to explore?
Ignoring changes to the model architecture, what might notably improve the prediction quality?


# Episode 4-advanced-layer-types.Rmd 

:: challenge
# Explore the data
amiliarize yourself with the CIFAR10 dataset. To start, consider the following questions:

. What is the dimension of a single data point? What do you think the dimensions mean?
. What is the range of values that your input data takes?
. What is the shape of the labels, and how many labels do we have?
. (Optional) We are going to build a new architecture from scratch to get you
amiliar with the convolutional neural network basics.
ut in the real world you wouldn't do that.
o the challenge is: Browse the web for (more) existing architectures or pre-trained models that are likely to work
ell on this type of data. Try to understand why they work well for this type of data.


:: challenge
# Number of parameters
uppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?


:: challenge
# Border pixels
hat, do you think, happens to the border pixels when applying a convolution?


:: challenge
# Number of model parameters
uppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise


:: challenge
# Convolutional Neural Network

nspect the network above:

What do you think is the function of the `Flatten` layer?
Which layer has the most parameters? Do you find this intuitive?
(optional) Pick a model from https://paperswithcode.com/sota/image-classification-on-cifar-10 . Try to understand how it works.


:: challenge
# Network depth
hat, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters?
ry it out. Create a `model` that has an additional `Conv2d` layer with 50 filters after the last MaxPooling2D layer. Train it for 20 epochs and plot the results.

*HINT**:
he model definition that we used previously needs to be adjusted as follows:
``python
nputs = keras.Input(shape=train_images.shape[1:])
= keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
= keras.layers.MaxPooling2D((2, 2))(x)
= keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
= keras.layers.MaxPooling2D((2, 2))(x)
Add your extra layer here
= keras.layers.Flatten()(x)
= keras.layers.Dense(50, activation='relu')(x)
utputs = keras.layers.Dense(10)(x)
``


:: challenge
# Why and when to use convolutional neural networks
. Would it make sense to train a convolutional neural network (CNN) on the penguins dataset and why?
. Would it make sense to train a CNN on the weather dataset and why?
. (Optional) Can you think of a different machine learning task that would benefit from a
NN architecture?


:: challenge
# Vary dropout rate
. What do you think would happen if you lower the dropout rate? Try it out, and
ee how it affects the model training.
. You are varying the dropout rate and checking its effect on the model performance,
hat is the term associated to this procedure?


