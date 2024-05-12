# -*- coding: utf-8 -*-

#importing libraries
import pandas as pd
import re #regular expression for handling texts very efficiently
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#loading data
df = pd.read_csv('path', delimiter='\t', header=0)
df.head()

#checking shape
df.shape

#checking distribution of Liked values
print(df['Liked'].value_counts())

#checking data types
df.dtypes

#checking for missing values
df.isnull().sum(axis = 0)


#cleaning the Reviews column using regular expressions
#removing punctuation and spaces; replacing special characters,<br /> in the file
df['Review'] = df['Review'].apply(lambda x: re.sub(r'[^A-Za-z0-9]+',' ',x))
df['Review'] = df['Review'].apply(lambda x: re.sub(r"<br />", " ", x))
# removing words with a length less than or equal to 2
df['Review'] = df['Review'].apply(lambda x: re.sub(r'\b[a-zA-Z]{1,2}\b', '', x))
df['Review'].head()

#performing train-test split
X = df['Review'].values
y = df['Liked'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

#printing shape
print(f'X_train size={X_train.shape}; X_test size  ={X_test.shape}')

#specifying the vocab size
VOCAB_SIZE = 1000
#performing textvectorization
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)

#fitting the state of the preprocessing layer to the dataset.
encoder.adapt(X_train)

# specifying dropout rate
dropout_rate = 0.3

#building RNN model
model_1 = tf.keras.Sequential([
    #performing textvectorization which converts the raw texts to indices/integers
    encoder,
    #embedding layer to convert the indices to numerical vectors
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=128,
        #using masking to handle the variable sequence lengths
        mask_zero=True),
    #GRU layer; the default recurrent_activation = sigmoid
    tf.keras.layers.GRU(256, return_sequences=False),
    #classification layer 1
    tf.keras.layers.Dense(128, activation='relu'),
    #dropout layer 1
    tf.keras.layers.Dropout(dropout_rate),
    #classification layer 2
    tf.keras.layers.Dense(64, activation='relu'),
    #classification layer 3; must be equal to 1 since this is the output layer
    tf.keras.layers.Dense(1, activation=None)
])


#summarizing model
model_1.summary()


#adding early stopping; if the validation accuracy does not improve for 7 epochs, we will stop training
callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience= 7)


#configuring the model; since activation=None we must put from_logits=True
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']) # we will evaluate the model using acc


# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history_1 = model_1.fit(x=X_train,
#                     y=y_train,
#                     epochs=100,
#                     validation_data=(X_test,y_test),
#                     callbacks=[callback],
#                     verbose = 1)


#visualizing training history
train_history = pd.DataFrame(history_1.history)
train_history['epoch'] = history_1.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()


#forecasting the Liked labels
#the cutoff probability is 50%
y_pred = (model_1.predict(X_test)> 0.5).astype(int)

#we need to convert 0 and 1 back to words so that 0 = Not Liked and 1 = Liked for easier readability
#printing a classification report
label_names = ['Not Liked', 'Liked']
report_1 = classification_report(y_test, y_pred, target_names=label_names)
print(report_1)


#building RNN model
model_2 = tf.keras.Sequential([
    #performing textvectorization which converts the raw texts to indices/integers
    encoder,
    #embedding layer to convert the indices to numerical vectors
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=128,
        #using masking to handle the variable sequence lengths
        mask_zero=True),
    #LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    #classification layer 1
    tf.keras.layers.Dense(128, activation='relu'),
    #dropout layer 1
    tf.keras.layers.Dropout(dropout_rate),
    #classification layer 2
    tf.keras.layers.Dense(64, activation='relu'),
    #classification layer 3; must be equal to 1 since this is the output layer
    tf.keras.layers.Dense(1, activation=None)
])


#summarizing model
model_2.summary()


#configuring the model; since activation=None we must put from_logits=True
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']) # we will evaluate the model using accuracy


# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history_2 = model_2.fit(x=X_train,
#                     y=y_train,
#                     epochs=100,
#                     validation_data=(X_test,y_test),
#                     callbacks=[callback],
#                     verbose = 1)


#visualizing training history
train_history = pd.DataFrame(history_2.history)
train_history['epoch'] = history_2.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()


#the cutoff probability is 50%
#forecasting the Recommended IND labels

y_pred = (model_2.predict(X_test)> 0.5).astype(int)

#printing classification report
report_2 = classification_report(y_test, y_pred, target_names=label_names)
print(report_2)


#building RNN model
model_3 = tf.keras.Sequential([
    #performing textvectorization which converts the raw texts to indices/integers
    encoder,
    #embedding layer to convert the indices to numerical vectors
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=128,
        #using masking to handle the variable sequence lengths
        mask_zero=True),
    #GRU layer; the default recurrent_activation = sigmoid
    tf.keras.layers.GRU(256, return_sequences=True),
    #LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
    #classification layer 1
    tf.keras.layers.Dense(128, activation='relu'),
    #dropout layer 1
    tf.keras.layers.Dropout(dropout_rate),
    #classification layer 2
    tf.keras.layers.Dense(64, activation='relu'),
    #classification layer 3; must be equal to 1 since this is the output layer
    tf.keras.layers.Dense(1, activation=None)
])

#summarizing model
model_3.summary()

#configuring the model; since activation=None we must put from_logits=True
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']) # we will evaluate the model using accuracy

# Commented out IPython magic to ensure Python compatibility.
# %%time
# #training the model and storing the history in a variable to plot later
# history_3 = model_3.fit(x=X_train,
#                     y=y_train,
#                     epochs=100,
#                     validation_data=(X_test,y_test),
#                     callbacks=[callback],
#                     verbose = 1)

#visualizing training history
train_history = pd.DataFrame(history_3.history)
train_history['epoch'] = history_3.epoch
#plotting train loss
sns.lineplot(x='epoch', y ='loss', data =train_history)
#plotting validation loss
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
#adding legends
plt.legend(labels=['train_loss', 'val_loss'])
plt.title('Training and Validation Loss Over Epochs')
plt.show()

#plotting training accuracy
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
#Plot validation accuracy
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
#adding legends
plt.legend(labels=['train_accuracy', 'val_accuracy'])
plt.title('Training and Validation Accuracy Over Epochs')
plt.show()
look at the classification report.


#the cutoff probability is 50%
#forecasting the Liked labels
y_pred = (model_3.predict(X_test)> 0.5).astype(int)

#printing a classification report
report_3 = classification_report(y_test, y_pred, target_names=label_names)
print(report_3)

