![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-08-21-ds-dl-intro day 1

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2023-08-21-ddi-day1)

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

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city


## 🥶🧊🍧 :ice_cream: :icecream: Icebreaker
What's your favourite breakfast?


## 🗓️ Agenda
|  Time |                              Topic |     |
| -----:| ----------------------------------:| --- |
|  9:00 |             Welcome and icebreaker |     |
|  9:15 |      Introduction to Deep Learning |     |
| 10:15 |                       Coffee break |     |
| 10:30 | Classification by a Neural Network |     |
| 11:30 |                       Coffee break |     |
| 11:45 | Classification by a Neural Network | :on:|
| 12:45 |                            Wrap-up |     |
| 13:00 |                                END |     |

## 🧠 Collaborative Notes & 🔧 Exercises

### Introduction to Deep Learning

[Visualization of a Neural Network](https://playground.tensorflow.org)

### 🔧 Exercise 1: Calculate the output for one neuron
Suppose we have:

- Input: X = (0, 0.5, 1)
- Weights: W = (-1, -0.5, 0.5)
- Bias: b = 1
- Activation function _relu_: `f(x) = max(x, 0)`

What is the output of the neuron?

_Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._

> **Solution** 
Weighted sum of input: 0 * (-1) + 0.5 * (-0.5) + 1 * 0.5 = 0.25
Add the bias: 0.25 + 1 = 1.25
Apply activation function: max(1.25, 0) = 1.25
So, neuron output = 1.25

### 🔧 Exercise 2: Mean Squared Error
One of the simplest loss functions is the Mean Squared Error. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$ .
It is the mean of all squared errors, where the error is the difference between the predicted and expected value.
In the following table, fill in the missing values in the 'squared error' column. What is the MSE loss for the predictions on these 4 samples?

| **Prediction** | **Expected value** | **Squared error** |
| -------------- | ------------------ | ----------------- |
| 1              | -1                 | 4                 |
| 2              | -1                 | ..                |
| 0              | 0                  | ..                |
| 3              | 2                  | ..                |
|                | **MSE:**           | ..                |

> **Solution**
 
| **Prediction** | **Expected value** | **Squared error** |
| -------------- | ------------------ | ----------------- |
| 1              | -1                 | 4                 |
| 2              | -1                 | 9                 |
| 0              | 0                  | 0                 |
| 3              | 2                  | 1                 |
|                | **MSE:**           | 3.5               |

_What sort of problems can Deep Learning solve?_
- Pattern/object recognition
- Segmenting images (or any data)
- Translating between one set of data and another, for example natural language translation.
- Generating new data that looks similar to the training data, often used to create synthetic datasets, art or even “deepfake” videos.

_What can Neural Networks not solve:_
- Any case where only a small amount of training data is available.
- Tasks requiring an explanation of how the answer was arrived at.
- Classifying things which are nothing like their training data.

_When is it overkill:_
- Logic operations, such as computing totals, averages, ranges etc.
- Modelling well defined systems, where the equations governing them are known and understood.
- Basic computer vision tasks such as edge detection, decreasing colour depth or blurring an image.

### 🔧 Exercise 3: Deep Learning Problems Exercise (in breakout rooms)

Which of the following would you apply Deep Learning to?

1. Recognising whether or not a picture contains a bird.
2. Calculating the median and interquartile range of a dataset.
3. Identifying MRI images of a rare disease when only one or two example images available for training.
4. Identifying people in pictures after being trained only on cats and dogs.
5. Translating English into French.

> **Solution**
> 1 and 5 are the sort of tasks often solved with Deep Learning.
2 is technically possible but solving this with Deep Learning would be extremely wasteful, you could do the same with much less computing power using traditional techniques.
3 will probably fail because there is not enough training data.
4 will fail because the Deep Learning system only knows what cats and dogs look like, it might accidentally classify the people as cats or dogs.

### Deep Learning workflow
1. Formulate/ Outline the problem
2. Identify inputs and output
3. Prepare data
4. Choose a pre-trained model or build a new architecture from scratch
5. Choose a loss function and optimizer
6. Train the model
7. Perform a Prediction/Classification -> test it!
8. Measure performance, with your selected metric
9. Tune Hyperparameters
10. Share Model e.g. hugging face

### Deep Learning Libraries
- [Tensorflow](https://www.tensorflow.org/) : Tensor calculations.
- [Pytorch](https://pytorch.org/)
- [Keras](https://keras.io/) : More user-friendly, will be using for this course.


### Testing your setup:
```bash
conda env list
conda activate dl_workshop
# open jupyter lab (jupyter notebooks)
jupyter lab
```
>Commands for jupyter notebooks:
Ctrl + Enter = execute, 
Shift + Enter = execute and go to next cell (create new if not available yet)

```python
from tensorflow import keras
# hit shift+enter to run a cell
import seaborn
print(seaborn.__version__)
# should show your version of seaborn (something like 0.12.2)
import sklearn
```

### Classification by a neural network using Keras

#### 1. Formulate/outline the problem: 
The goal is to predict a penguins’ species using the attributes available in this dataset.
#### 2. Identify inputs and outputs
```python
import seaborn as sns
# Load a pandas dataframe
penguins = sns.load_dataset('penguins')
# explore the dataset
penguins.head() # shows first lines of the dataset
penguins.shape # check the amount of data/examples that we have available
# We can plot the data:
sns.pairplot(penguins, hue="species")
```
- __Input__: `bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g`
- __Output__: `species`

### 🔧 Exercise 4: Pairplot
Take a look at the pairplot we created. Consider the following questions:

- Is there any class that is easily distinguishable from the others?
- Which combination of attributes shows the best separation for all 3 class labels at once?

>**Solution**
>The plots show that the green class, Gentoo is somewhat more easily distinguishable from the other two. The other two seem to be separable by a combination of bill length and bill depth (other combinations are also possible such as bill length and flipper length).

#### 3. Prepare the dataset

```python
penguins_filtered = penguins.drop(columns=['island', 'sex'])
# drop some missing data
penguins_filtered = penguins_filtered.dropna()
# Extract columns corresponding to features
penguins_features = penguins_filtered.drop(columns=['species'])
```
Prepare the data for training:
How to handle string data? $\to$ convert them to numerical values using **one-hot encoding**
```python
import pandas as pd
target = pd.get_dummies(penguins_filtered['species'])
target.head() # print out the top 5 to see what it looks like.
```

## 🔧 Exercise 5: One-hot encoding vs ordinal encoding
How many output neurons will our network have now that we one-hot encoded the target class?
- A: 1
- B: 2
- C: 3 :heavy_check_mark:

Split data into training and test set
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(penguins_features, target, test_size=0.2, random_state=0, shuffle=True, stratify=target)
```
- `test_size`: fraction of the dataset that would be put in the test dataset
- `random_state`: controls the shuffling of the dataset, setting this value will reproduce the same results (assuming you give the same integer) every time it is called.
- `shuffle=True`: The rows will be shuffled before splitting the dataset.
- `stratify=target`: train and test sets the function will return will have roughly the same proportions (with regards to the number of penguins of a certain species) as the dataset.

#### 4. Build an architecture from scratch or choose a pretrained model

```python
from tensorflow import keras
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
```
Building our Neural network: defining the architecture
```python=
inputs = keras.Input(shape=X_train.shape[1])
hidden_layer = keras.layers.Dense(10, activation="relu")(inputs)
```
- `Dense`: type of layer in whcih all neurons are conected to all neurons in the previous layer.
- `10` in this case is teh number of neurons, this is a hyperparameter.
- `activation="relu"` is 0 for inputs that are 0 and below and the identity function (returning the same value) for inputs above 0.
- `(inputs)` This tells the Dense layer to connect the layer passed as a parameter, in this case the inputs.

```python=
output_layer = keras.layers.Dense(3, activation="softmax")(hidden_layer)
```
- `activation="softmax"` This activation function ensures that the three output neurons produce values in the range (0, 1) and they sum to 1. We can interpret this as a kind of ‘probability’ that the sample belongs to a certain species.

Now that we have all the elements (layers) we can put together our model:
```python=
model = keras.Model(inputs=inputs, outputs=output_layer)
# and we can get a nice summary of our model:
model.summary()
```

Keras distinguishes between two types of weights, namely:

- **trainable parameters**: these are weights of the neurons that are modified when we train the model in order to minimize our loss function (we will learn about loss functions shortly!).

- **non-trainable parameters**: these are weights of the neurons that are not changed when we train the model. These could be for many reasons - using a pre-trained model, choice of a particular filter for a convolutional neural network, and statistical weights for batch normalization are some examples.

### 🔧 Exercise 6: Create the neural network
With the code snippets above, we defined a Keras model with 1 hidden layer with
10 neurons and an output layer with 3 neurons.

* How many parameters does the resulting model have?
* What happens to the number of parameters if we increase or decrease the number of neurons in the hidden layer?

>Solution:
>The model has 83 trainable parameters. If you increase the number of neurons in the hidden layer the number of trainable parameters in both the hidden and output layer increases or decreases accordingly of neurons.

#### 5. Choose a loss function and optimizer

- [Loss functions in Keras](https://www.tensorflow.org/api_docs/python/tf/keras/losses): we will use Categorical Crossentropy because it works by comparing the probabilities that the neural network predicts with ‘true’ probabilities that we generated using the one-hot encoding.
- Optimizer: We will use **Adam** that is widely used.

```python
model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
```

#### 6. Train model

```python
history = model.fit(X_train, y_train, epochs=100)
```
- One training epoch means that every sample in the training data has been shown to the neural network and used to update its parameters.

We can plot the results of the training stored in the object `history`:
```python
sns.lineplot(x=history.epoch, y=history.history['loss'])
```

### 🔧 Exercise 7: The Training Curve
Looking at the training curve we have just made.

1. How does the training progress?
   * Does the training loss increase or decrease?
   * Does it change quickly or slowly?
   * Does the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?

When the training process does not go well:

3. (optional) Something went wrong here during training. What could be the problem, and how do you see that in the training curve?
Also compare the range on the y-axis with the previous training curve.
![](https://codimd.carpentries.org/uploads/upload_9a7323086d120cf1de9be691a79ca64e.png)


## 📚 Resources

- Example of explainable AI software: [DIANNA](https://github.com/dianna-ai/dianna)
- [Hugging Face](https://huggingface.co/)
- R Libraries:
    - [Tensorflow for R](https://tensorflow.rstudio.com/)
    - [torch for R](https://torch.mlverse.org/)
    - [Keras for R](https://cran.r-project.org/web/packages/keras/vignettes/)
- [Google tuning playbook](https://github.com/google-research/tuning_playbook): A guide on how to choose your hyperparameters.
- [Blog: Activation functions](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)
- [Tensorboard](https://www.tensorflow.org/tensorboard)
- [Layers in Keras](https://keras.io/api/layers/)
- [Optimizers in Keras](https://keras.io/api/optimizers/)
- [Loss functions in Keras](https://keras.io/api/losses/)