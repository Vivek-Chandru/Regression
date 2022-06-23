#!/usr/bin/env python
# coding: utf-8

# # 1. Vorverarbeitung
# Laden Sie die gegebenen Trainings- und Testdaten von Cloudserver herunter. Visualisieren Sie die Daten. Teilen Sie die gegebenen Trainingsdaten geeignet in eine Trainings- und Validierungsmenge auf. Fassen sie die Daten jeweils in einem Datensatz zusammen.

# ## Imports

# In[62]:


import numpy as np
import matplotlib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import time
from keras.callbacks import EarlyStopping
import matplotlib.backends.backend_pdf


# ## Read Data:
# 
# - extracts the RAR file
# - reads the CSV data
# - creates a list of test and trainig dataframes

# In[63]:


def read_data(train_path, test_path):
    data_test = []
    data_train = []
    for item in os.listdir(train_path):
        item = train_path + '/'+item
        content = pd.read_csv(item, header=0)
        data_train.append(content)
    for item in os.listdir(test_path):
        item = test_path + '/'+item
        f=open(item, 'r')
        content = pd.read_csv(f, header=0)
        data_test.append(content)
    return (data_train,data_test)


# In[64]:


Train_path= r"/home/fpds01/work/final_projekt/Training/Files_IAV/Training"
Test_path = r"/home/fpds01/work/final_projekt/Training/Files_IAV/Test"
image_path = r"/home/fpds01/work/final_projekt/Training/Final_Notebooks/MSE_images"


# In[65]:


(data_train, data_test) = read_data(Train_path, Test_path)


# ## Data Exploration:
#  - resets index of the dataframes in the data list
#  - plots the variables and target over time for the first and second dataframe

# In[66]:


data_train[0].plot(subplots=True, figsize=(15, 10))
plt.savefig(image_path+"/data_plots.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[67]:


def reset_index(ds, newindex='time'):
    for item in ds:
        item.set_index(newindex, inplace=True)
reset_index(data_train)
reset_index(data_test)


# ## Data Consolidation:
# - The lists of training and testing dataframes are converted into one dataframe, while retaining the column labels

# In[68]:


def consolidate(data_list):
    data_cons = data_list[0]
    for i in range(len(data_list)-1):
        data_cons =data_cons.append(data_list[i+1])
    return data_cons


# In[69]:


data_train = consolidate(data_train)
data_test = consolidate(data_test)


# ## Data Normalizing:
# - Variable entries in the data frame are min-max normalized with respect to min and max values of each variable.

# In[70]:


scaler = MinMaxScaler()
data_train = pd.DataFrame(scaler.fit_transform(data_train), columns=data_train.columns)
data_test = pd.DataFrame(scaler.transform(data_test), columns=data_test.columns)


# ## Data split train and validation

# In[71]:


data_train, data_val = train_test_split(data_train, train_size = 0.9)
data_train_size = data_train.size
data_val_size = data_val.size

print("Trainig data size = ",data_train_size)
print("Validation data size = ",data_val_size)


# ## Dataframe to tf dataset conversion:
# 
# The normalized pandas dataframes are converted to tensorflow datasets to take advantage of tensorflow functions

# In[72]:


def convert_to_tfds(data_frame):
    target = data_frame.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((data_frame.values, target.values))
    return dataset
        


# In[73]:


train_ds = convert_to_tfds(data_train)
test_ds = convert_to_tfds(data_test)
val_ds = convert_to_tfds(data_val)


# ## Making an Input Pipeline

# In[74]:


batch_size = 64
train_ds = train_ds.repeat().shuffle(buffer_size =data_train_size,seed =0).batch(batch_size).prefetch(buffer_size =1)
test_ds = test_ds.batch(1).prefetch(buffer_size =1)
val_ds = val_ds.repeat().batch(batch_size).prefetch(buffer_size =1)


# # 2. Model 

# In[75]:


model = keras.Sequential(
    [
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ]
)


# In[76]:


def MSE_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1) 


# In[77]:

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=MSE_loss_fn)


# In[78]:


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.timetaken = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append((epoch,time.process_time() - self.timetaken))
    def on_train_end(self,logs = {}):
        plt.figure()
        print("Training time = ",self.times)
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(*zip(*self.times))
        plt.savefig(image_path+"/MSE_timing_plot.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[79]:


es = EarlyStopping(monitor='val_loss',patience=10)


# In[80]:


timetaken = timecallback()
history = model.fit(train_ds,
             epochs=150,
             steps_per_epoch=(data_train_size/batch_size)-1,
             validation_data=val_ds,
             validation_steps=(data_val_size/batch_size)-1, verbose  = 0,callbacks = [timetaken,es])


# In[81]:

np.save("MSE_history.npy",history.history)
model.summary()

file_name = "MSE_model.h5"
model.save(file_name)

# In[82]:


plt.figure()
sns.set()
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})
history_frame = pd.DataFrame(history.history) 
history_frame.plot()
plt.savefig(image_path+"/MSE_training_plot.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[83]:


y_pred_test = model.predict(test_ds)


# In[89]:


y_true_test = [item[1] for item in test_ds]
y_true_test = np.array(y_true_test)
plt.figure(figsize=(30, 5))
plt.plot( y_pred_test[0:100], marker='o', markerfacecolor='blue', markersize=5, color='skyblue', linewidth=1)
plt.plot( y_true_test[0:100], marker='', color='olive', linewidth=1)
plt.legend(['Predicted Value','True Value'])
plt.xlabel('Test Samples')
plt.savefig(image_path+"/MSE_result_plots.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')



# In[87]:


mse = tf.keras.losses.MeanSquaredError()
print("MSE on test data =",mse(y_true_test, y_pred_test).numpy())


# In[88]:


mae = tf.keras.losses.MeanAbsoluteError()
print("MAE on test data =",mae(y_true_test, y_pred_test).numpy())

