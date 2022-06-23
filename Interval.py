#!/usr/bin/env python
# coding: utf-8

# # 1. Vorverarbeitung
# Laden Sie die gegebenen Trainings- und Testdaten von Cloudserver herunter. Visualisieren Sie die Daten. Teilen Sie die gegebenen Trainingsdaten geeignet in eine Trainings- und Validierungsmenge auf. Fassen sie die Daten jeweils in einem Datensatz zusammen.

# ## Imports

# In[1]:


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

# In[2]:


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


# In[3]:


Train_path= r"/home/fpds01/work/final_projekt/Training/Files_IAV/Training"
Test_path = r"/home/fpds01/work/final_projekt/Training/Files_IAV/Test"
image_path = r"/home/fpds01/work/final_projekt/Training/Final_Notebooks/INT_images"


# In[4]:


(data_train, data_test) = read_data(Train_path, Test_path)


# ## Data Exploration:
#  - resets index of the dataframes in the data list
#  - plots the variables and target over time for the first and second dataframe

# In[5]:


data_train[0].plot(subplots=True, figsize=(15, 10))
plt.savefig(image_path+"/data_plots.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[6]:


def reset_index(ds, newindex='time'):
    for item in ds:
        item.set_index(newindex, inplace=True)
reset_index(data_train)
reset_index(data_test)


# ## Data Consolidation:
# - The lists of training and testing dataframes are converted into one dataframe, while retaining the column labels

# In[7]:


def consolidate(data_list):
    data_cons = data_list[0]
    for i in range(len(data_list)-1):
        data_cons =data_cons.append(data_list[i+1])
    return data_cons


# In[8]:


data_train = consolidate(data_train)
data_test = consolidate(data_test)


# ## Data Normalizing:
# - Variable entries in the data frame are min-max normalized with respect to min and max values of each variable.

# In[9]:


scaler = MinMaxScaler()
data_train = pd.DataFrame(scaler.fit_transform(data_train), columns=data_train.columns)
data_test = pd.DataFrame(scaler.transform(data_test), columns=data_test.columns)


# ## Data split train and validation

# In[10]:


data_train, data_val = train_test_split(data_train, train_size = 0.9)
data_train_size = data_train.size
data_val_size = data_val.size

print("Trainig data size = ",data_train_size)
print("Validation data size = ",data_val_size)


# ## Dataframe to tf dataset conversion:
# 
# The normalized pandas dataframes are converted to tensorflow datasets to take advantage of tensorflow functions

# In[11]:


def convert_to_tfds(data_frame):
    target = data_frame.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((data_frame.values, target.values))
    return dataset
        


# In[12]:


train_ds = convert_to_tfds(data_train)
test_ds = convert_to_tfds(data_test)
val_ds = convert_to_tfds(data_val)


# ## Making an Input Pipeline

# In[13]:


batch_size = 64
train_ds = train_ds.repeat().shuffle(buffer_size =data_train_size,seed =0).batch(batch_size).prefetch(buffer_size =1)
test_ds = test_ds.batch(1).prefetch(buffer_size =1)
val_ds = val_ds.repeat().batch(batch_size).prefetch(buffer_size =1)


# # 2. Model 

# In[14]:


model = keras.Sequential(
    [
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)
    ]
)


# In[15]:


def INT_loss_fn(y_true, y_pred):
    beta = 0.00001
    print(y_pred)
    y_upper = tf.maximum(tf.gather(y_pred,[0],axis = 1),tf.gather(y_pred,[1],axis = 1))
    y_lower = tf.minimum(tf.gather(y_pred,[0],axis = 1),tf.gather(y_pred,[1],axis = 1))
    zero = tf.zeros(shape = (batch_size,1), dtype = tf.dtypes.float32)
    loss = tf.square(tf.maximum(y_true-y_upper, zero))+tf.square(tf.maximum(y_lower-y_true,zero))+ beta*(y_upper-y_lower)
    return tf.reduce_mean(loss, axis = -1) 


# In[16]:

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=INT_loss_fn)


# In[17]:


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
        plt.savefig(image_path+"/Interval_timing_plot.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[18]:


es = EarlyStopping(monitor='val_loss',patience=10)


# In[19]:


timetaken = timecallback()
history = model.fit(train_ds,
             epochs=150,
             steps_per_epoch=(data_train_size/batch_size)-1,
             validation_data=val_ds,
             validation_steps=(data_val_size/batch_size)-1, verbose  = 0,callbacks = [timetaken,es])


# In[20]:

np.save("INT_history.npy",history.history)
model.summary()

file_name = "INT_model.h5"
model.save(file_name)

# In[21]:


plt.figure()
sns.set()
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})
history_frame = pd.DataFrame(history.history) 
history_frame.plot()
plt.savefig(image_path+"/Interval_training_plot.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[22]:


y_pred = model.predict(test_ds)


# In[23]:


y_true = [item[1] for item in test_ds]


# In[24]:


y_pred_upper = tf.maximum(y_pred[:,0],y_pred[:,1])
y_pred_lower = tf.minimum(y_pred[:,0],y_pred[:,1])
y_true = np.array(y_true)
x_plot = list(range(100))
plt.figure(figsize=(30, 5))
plt.plot(x_plot, y_true[0:100], marker='', color='olive', linewidth=3)
plt.fill_between(x_plot, y_pred_upper[0:100], y_pred_lower[0:100],color='yellow')
plt.legend(['True Value','Confidence interval'])
plt.xlabel('Test Samples')
plt.savefig(image_path+"/Interval_result_plots.png", bbox_inches='tight', facecolor = 'white', edgecolor = 'white')


# In[26]:


y_true.flatten()
y_true = tf.convert_to_tensor(y_true, np.float32)


# In[27]:


mse = tf.keras.losses.MeanSquaredError()
print("MSE for upper limit on test data =",mse(y_true, y_pred_upper).numpy())
print("MSE for lower limit on test data =",mse(y_true, y_pred_lower).numpy())


# In[28]:


mae = tf.keras.losses.MeanAbsoluteError()
print("MAE for upper limit on test data =",mae(y_true, y_pred_upper).numpy())
print("MAE for lower limit on test data =",mae(y_true, y_pred_lower).numpy())

