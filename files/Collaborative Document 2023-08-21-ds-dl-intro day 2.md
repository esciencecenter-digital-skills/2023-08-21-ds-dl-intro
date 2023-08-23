![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-08-21-ds-dl-intro day 2

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: [link](https://tinyurl.com/2023-08-21-ddi-day2)

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
What is your favourite zoom background? (Show us!)

## 🗓️ Agenda
|  Time |                              Topic |
| -----:| ----------------------------------:|
|  9:00 |             Welcome and icebreaker |
|  9:15 | Classification by a Neural Network |
| 10:15 |                       Coffee break |
| 10:30 | Classification by a Neural Network |
| 11:30 |                       Coffee break |
| 11:45 |       Monitor the training process | 
| 12:45 |                            Wrap-up |
| 13:00 |                                END |

## 🧠 Collaborative Notes & 🔧 Exercises

Correction from yesterday to ensure consistent results when you rerun you notebook. This is at the part where we started to build a network.
```python
# instead of using:
# from tensorflow.random import set_seed
# set_seed(2)
# use:
keras.utils.set_random_seed(2)
```
#### Step 7: perform a prediction
```python
y_pred = model.predict(X_test)
```


```python
prediction = pd.DataFrame(y_pred, columns=target.columns)
prediction.head()
```

```python
predicted_species = prediction.idxmax(axis="columns")
predicted_species.head()
```

#### Step 8: Measuring preformance
```python
from sklearn.metrics import confusion_matrix
true_species = y_test.idxmax(axis="columns")
```

```python
true_species.head()
```

```python
matrix = confusion_matrix(true_species, predicted_species)
matrix
```

```python
confusion_df = pd.DataFrame(
    matrix, index=y_test.columns.values,
    columns=y_test.columns.values)
```

```python
sns.heatmap(confusion_df, annot=True)
```

### Exercise 1
Measure the performance of the neural network you trained and visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?

#### Step 9: Tune hyperparameters

#### Step 10:
```python
model.save('my_first_cute_penguin_model')
```

```python
# load your saved model
loaded_model = keras.models.load_model('my_first_cute_penguin_model')
```

#### Key points
1. Use the workflow to structure your work
2. You now know how to create a model from scratch in keras
3. First take a lot of shortcuts, get your pipeline up and running, then yuo start adding new stuff.

### Monitor the training process

1. Regression task
2. Optimization: gradient descent
3. Overfitting
4. Improve the model performance


#### 1. Outline the problem
Predict the sunshine hours for tomorrow for Basel using the data from today.

#### 2. Identify inputs and outputs
```python
import pandas as pd

filename_data = "weather_prediction_dataset_light.csv"
```

```python
data = pd.read_csv(filename_data)
data.head()
```

If you have not downloaded the data yet, you can also load it directly from Zenodo:
```python
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
```

```python
data.columns
```

```python
data.shape
```

#### Prepare the data
```python
# We will only use the first three year of data for now 
nr_rows = 365*3

X_data = data.loc[:nr_rows]
X_data = X_data.drop(columns=['DATE', 'MONTH'])
```

```python
y_data = data.loc[1:(nr_rows+1)]["BASEL_sunshine"]
```

There are no NaN values in this data, so we can skip the step to drop the rows containing NaN values.

Split data into a training, validation, and test set. A validation data set is used for hyperparameter tuning. Only when you are happy with the hyperparameter settings, you use the test set.
```python
from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
```

```python
X_holdout.shape
```

Split the holdout data in a validation and test set:
```python
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout, y_holdout, 
    test_size=0.5, random_state=0)
```

### Exercise 2: architecture of the network
As we want to design a neural network architecture for a regression task, see if you can first come up with the answers to the following questions:

1. What must be the dimension of our input layer?
2. We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the softmax activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression? Hint: A layer with relu activation, with sigmoid activation or no activation at all?
3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in addition to the sunshine hours?

#### 4. Build the architecture
```python
from tensorflow import keras

def create_nn():
    # Input layer
    inputs = keras.Input(shape=(X_data.shape[1]), name='input')
    
    # Hidden dense layer
    layers_dense_1 = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense_2 = keras.layers.Dense(50, 'relu')(layers_dense_1)
    
    # Output layer, we do not use an activation function for this
    outputs = keras.layers.Dense(1)(layers_dense_2)
    
    return keras.Model(inputs=inputs, outputs=outputs,
                       name='weather_prediction_model')
```

```python
model = create_nn()
```

```python
model.summary()
```

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/03_gradient_descent.png)


### Exercise 3: gradient descent
Answer the following questions:

1. What is the goal of optimization?
A. To find the weights that maximize the loss function
B. To find the weights that minimize the loss function
2. What happens in one gradient descent step?
A. The weights are adjusted so that we move in the direction of the gradient, so up the slope of the loss function
B. The weights are adjusted so that we move in the direction of the gradient, so down the slope of the loss function
C. The weights are adjusted so that we move in the direction of the negative gradient, so up the slope of the loss function
D. The weights are adjusted so that we move in the direction of the negative gradient, so down the slope of the loss function
3. When the batch size is increased:
(multiple answers might apply)

A. The number of samples in an epoch also increases
B. The number of batches in an epoch goes down
C. The training progress is more jumpy, because more samples are consulted in each update step (one batch).
D. The memory load (memory as in computer hardware) of the training process is increased

#### 5. Choose a loss function and optimizer
```python
def compile_model(model):
    model.compile(loss='mse', optimizer='adam',
                  metrics=[keras.metrics.RootMeanSquaredError()])
    
compile_model(model)
```

#### 6. Train the model

```python
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    verbose=2)
```

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metrics")
    
plot_history(history, 'root_mean_squared_error')
```

#### 7. Perform a prediction
```python
y_test_predicted = model.predict(X_test)
y_train_predicted = model.predict(X_train)
```

#### 8. Measure performance
```python
def plot_predictions(y_pred, y_true, title):
    plt.style.use('ggplot')
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel('predicted sunshine hours')
    plt.ylabel('True sunshine hours')
    plt.title(title)
```
```python
plot_predictions(y_train_predicted, y_train,
                 title='Predictions on the train  set')
```

```python
plot_predictions(y_test_predicted, y_test,
                 title='Predictions on the test set')
```

### Exercise 4: Reflecting on our results
- Is the performance of the model as you expected (or better/worse)?
- Is there a noteable difference between training set and test set? And if so, any idea why?
- (Optional) When developing a model, you will often vary different aspects of your model like which features you use, model parameters and architecture. It is important to settle on a single-number evaluation metric to compare your models.
    - What single-number evaluation metric would you choose here and why?

#### Solution to optional exercise:
The metric that we are using: RMSE would be a good one. You could also consider Mean Squared Error, that punishes large errors more (because large errors create even larger squared errors). It is important that if the model improves in performance on the basis of this metric then that should also lead you a step closer to reaching your goal: to predict tomorrow’s sunshine hours. If you feel that improving the metric does not lead you closer to your goal, then it would be better to choose a different metric

```python
train_metrics = model.evaluate(X_train, y_train, return_dict=True)
test_metrics = model.evaluate(X_test, y_test, return_dict=True)
print('Train metrics', train_metrics)
print('Test metrics', test_metrics)
```

```python
y_baseline_prediction = X_test['BASEL_sunshine']
plot_predictions(y_baseline_prediction, y_test,
                title='Baseline predictions on the test set')
```

```python
from sklearn.metrics import mean_squared_error
rmse_baseline = mean_squared_error(y_test, y_baseline_prediction,
                                  squared=False)
```

```python
rmse_baseline
```

#### 9. Refine the model

```python
model = create_nn()
compile_model(model)
```

```python
history = model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=200,
                   validation_data=(X_val, y_val))
```

```python
plot_history(history, ['root_mean_squared_error',
                       'val_root_mean_squared_error'])
```

### Exercise 5: plot the training progress
1. Is there a difference between the training curves of training versus validation data? And if so, what would this imply?
2. (Optional) Take a pen and paper, draw the perfect training and validation curves. (This may seem trivial, but it will trigger you to think about what you actually would like to see)

#### Solution:
1. The difference in the two curves shows that something is not completely right here. The error for the model predictions on the validation set quickly seem to reach a plateau while the error on the training set keeps decreasing. That is a common signature of **overfitting**.
2. Ideally you would like the training and validation curves to be identical and slope down steeply to 0. After that the curves will just consistently stay at 0. 


## 📚 Resources
- [How to choose loss functions?](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
- [How to choose an optimization algorithm?](https://machinelearningmastery.com/tour-of-optimization-algorithms/)
- If you want to learn more about privacy-preserving AI: watch [this video](https://www.youtube.com/watch?v=4zrU54VIK6k) from one of my deep learning heroes Andrew Trask. Privacy-preserving AI is a form of machine learning/deep learning where you use a model that preserves privacy of the dataset.
- [ONNX](https://onnx.ai/) is a format that allos to share ML models (even between differnt frameworks: such as pytorch and keras).