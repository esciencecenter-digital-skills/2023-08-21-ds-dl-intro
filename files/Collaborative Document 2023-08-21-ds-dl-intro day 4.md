![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document 2023-08-21-ds-dl-intro day 4

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------
This is the Document for today: [link](https://tinyurl.com/2023-08-21-ddi-day4)

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
Sven's special - on popular request! 

## 🗓️ Agenda
|  Time |                  Topic |
| -----:| ----------------------:|
|  9:00 | Welcome and icebreaker |
|  9:15 |    Advance Layer Types |
| 10:15 |           Coffee break |
| 10:30 |    Advance Layer Types |
| 11:30 |           Coffee break |
| 11:45 |                Outlook | 
| 12:45 |                Wrap-up |
| 13:00 |                    END |

## 🧠 Collaborative Notes & 🔧 Exercises
#### Tips for structuring your own DL project
- Create a folder called notebooks
- Create numbered notebooks with one experiment per notebook. Give your notebooks sensible names
- Start your notebook with a rationale -> describe the experiment
- use all notebook cells to structure your experiment
- use pictures 
- Structure your notebook from top to bottom, so that you are able to rerun your full notebook.
- End with a conclusion
- Review each others notebook

#### Additional tips
- [Cookiecutter Data Science](http://drivendata.github.io/cookiecutter-data-science/) can help you organise your project. It has a some nice tips and tricks.
- have source code folder, in which you save repetitive code, such as functions you use in multiple experiments
- There is another eScience workshop on good practices in research software development


### Tune hyperparameters

```python
# check what the shape looks like
train_images.shape
train_images.shape[1:]
```


```python
# create out model
inputs = keras.Input(shape=train_images.shape[1:])

x = keras.layers.Conv2D(50, (3,3), activation='relu')(inputs)
x = keras.layers.Conv2D(50, (3,3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs,
                   name="cifar_model_small")

model.summary()



### Exercise 1: Convolutional Neural Network
1) What do you think is the function of the `Flatten` layer?
2) Which layer has the most parameters? Do you find this intuitive?
3) (optional) Pick a model from https://paperswithcode.com/sota/image-classification-on-cifar-10 . Try to understand how it works.

### 4. Chosse a pretrained model or start building architecture from scratch
To look for pretrained models, search for: <i> cifar10 state of the art keras</i> (for example)

```python
def create_nn():
    inputs = keras.Input(shape=train_images.shape[1:])

    x = keras.layers.Conv2D(50, (3,3), activation='relu')(inputs)
    # new layer
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(50, (3,3), activation='relu')(x)
    # new layer
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Flatten()(x)
    # new layer
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                       name="cifar_model")
    return model

model = create_nn()
model.summary()
```

#### 5. Choose a loss function and optimizer

```python
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    
compile_model(model)
```

#### 6. Train the model
```python
history = model.fit(train_images, train_labels, epochs=10,
                   validation_data=(test_images, test_labels))
```


```python

```

#### 7. Perform a prediction/classification
skip

#### 8. Measure performance

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

```python
def plot_history(history, metrics):
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")

plot_history(history, ['accuracy', 'val_accuracy'])
```

```python
plot_history(history, ['loss', 'val_loss'])
```
It seems that the model is overfitting somewhat, because the validation accuracy and loss stagnates.


### Exercise 2 Network depth
What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters? Try it out. Create a model that has an additional Conv2d layer with 50 filters after the last MaxPooling2D layer. Train it for 20 epochs and plot the results.


#### Solution
```python
def create_nn_extra_layer():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x) #
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x) # estra layer
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x) # a new Dense layer
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model

model = create_nn_extra_layer()
```

```python
model.summary()
```

```python
# compile and plot:
compile_model(model)
history = model.fit(train_images, train_labels, epochs=20,
                   validation_data=(test_images, test_labels))
plot_history(history, ['accuracy', 'val_accuracy'])
```


### Exercise 3: WHY AND WHEN TO USE CONVOLUTIONAL NEURAL NETWORKS
1. Would it make sense to train a convolutional neural network (CNN) on the penguins dataset and why?
2. Would it make sense to train a CNN on the weather dataset and why?
3. (Optional) Can you think of a different machine learning task that would benefit from a CNN architecture?


### Dropout layers
![](https://codimd.carpentries.org/uploads/upload_47ba554e0dd67cdc20fdbdd016e90526.png)


```python
def create_nn_with_dropout():
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x) #
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x) # estra layer
    # new dropout layer
    x = keras.layers.Dropout(0.8)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x) # a new Dense layer
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model
```

```python
model_dropout = create_nn_with_dropout(
model_dropout.summary())
```

```python
compile_model(model_dropout)
```

```python
history = model_dropout.fit(train_images, train_labels,
                           epochs=20,
                           validation_data = (test_images, test_labels))
```

```python
plot_history(history, ['accuracy', 'val_accuracy'])
```

```python
test_loss, test_acc = model_dropout.evaluate(test_images, test_labels, verbose=2)
```

```python
plot_history(history, ['loss', 'val_loss'])
```

### Exercise 4: Varying the dropout rate
1. What do you think would happen if you lower the dropout rate? Try it out, and see how it affects the model training.
2. You are varying the dropout rate and checking its effect on the model performance, what is the term associated to this procedure?

#### Solution:
```python
def create_nn_with_dropout(dropout_rate):
    inputs = keras.Input(shape=train_images.shape[1:])
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(50, activation='relu')(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")
    return model

dropout_rates = [0.15, 0.3, 0.45, 0.6, 0.75]
test_losses = []
for dropout_rate in dropout_rates:
    model_dropout = create_nn_with_dropout(dropout_rate)
    compile_model(model_dropout)
    model_dropout.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

    test_loss, test_acc = model_dropout.evaluate(test_images,  test_labels)
    test_losses.append(test_loss)

loss_df = pd.DataFrame({'dropout_rate': dropout_rates, 'test_loss': test_losses})

sns.lineplot(data=loss_df, x='dropout_rate', y='test_loss')
```

### 10. Save the model
```python
model.save("cc_model")
```
## Outlook: real-world application
[notebook](https://github.com/matchms/ms2deepscore/blob/0.4.0/notebooks/MS2DeepScore_tutorial.ipynb)
[Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4)

### Exercise 5: A REAL-WORLD DEEP LEARNING APPLICATION
1. Looking at the ‘Model training’ section of the notebook, what do you recognize from what you learned in this course?
2. Can you identify the different steps of the deep learning workflow in this notebook?
3. (Optional): Try to fully understand the neural network architecture from the first figure of the paper

#### Topics in real-world examples:
- Neural Networks
- early stopping
- model architecture
- Types of layers
- Adam optimizer
- dropout
- batch normalization
- saving the model
- non-trainable parameters
- hyperparameter tuning
- loss function, accuracy , metric
- metrics - rmse
- learning rate
- training, validation and test sets
- model.fit
- dense layers
- confusion matrix
- Model prediction step

#### Topics not seen in real-world example:
- Convolution layers
- problem definition
- different data-types
- softmax or activation functions
- pooling layers
- regression tasks
- handling missing data (data preparation)
- train and test split
- baselines


## 📚 Resources
- [Cookiecutter Data Science](http://drivendata.github.io/cookiecutter-data-science/)
- [eScience Center workshop (past event)](https://www.esciencecenter.nl/event/good-practices-in-research-software-development/)
- [CNN explainer](https://poloclub.github.io/cnn-explainer/): A nice interactive visualization of a convulotional neural network.
- [CIFAR-10 SOTA keras implementation](https://github.com/Adeel-Intizar/CIFAR-10-State-of-the-art-Model/blob/master/CIFAR-10%20Best.ipynb)
- [Real-world example](https://github.com/matchms/ms2deepscore/blob/0.4.0/notebooks/MS2DeepScore_tutorial.ipynb)
- [MS2DeepScore](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00558-4)
- [Link to learning resources](https://carpentries-incubator.github.io/deep-learning-intro/instructor/reference.html#external-references)
- [Google Colab](https://colab.google/) to get free access to GPU
- [GPT-4 explaing GPT-4](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)
