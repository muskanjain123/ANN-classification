import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
data = pd.read_csv("Churn_Modelling.csv")
data= data.drop(['RowNumber','CustomerId','Surname'],axis=1)

LabelEncoder_gender = LabelEncoder()
data['Gender'] = LabelEncoder_gender.fit_transform(data['Gender'])
print(data)

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder_geo=OneHotEncoder()
geo_encoder=one_hot_encoder_geo.fit_transform(data[['Geography']]).toarray()
print(geo_encoder)
print(one_hot_encoder_geo.get_feature_names_out(['Geography']))
geo_encode_df=pd.DataFrame(geo_encoder,columns=[one_hot_encoder_geo.get_feature_names_out(['Geography'])])
print(geo_encode_df)

data = pd.concat([data.drop('Geography',axis=1),geo_encode_df],axis=1)
print(data.head())


with open('labelEncoder_gender.pkl','wb') as file:
  pickle.dump(LabelEncoder_gender,file)

with open('one_hot_encoder_geo.pkl','wb') as file:
  pickle.dump(one_hot_encoder_geo,file)


X=data.drop('Exited',axis=1)
y=data['Exited']

## Split the data in training and tetsing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
## Scale these features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

print((X_train.shape[1],))

with open('scaler.pkl','wb') as file:
  pickle.dump(scaler,file)

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

model = Sequential([
  Dense(64,activation='relu',input_shape=(X_train.shape[1],)),
  Dense(32,activation='relu'),
  Dense(1,activation='sigmoid')  
]
)
print(model.summary())
import tensorflow

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

early_stopping_callbacks=EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callbacks]
)

model.save('my_model.keras')  # Save in the new Keras format

import webbrowser
from tensorboard import program

# Create a TensorBoard instance
tb = program.TensorBoard()

# Specify the log directory (change this path to your logs directory)
log_dir = './logs/fit2025-01-14-02-03-43'  # This should be your log directory path

# Optionally, specify a different port if needed (default is 6006)
#port = '6006'

# Configure TensorBoard to use the log directory
tb.configure(argv=[None, '--logdir', log_dir])

# Launch TensorBoard
url = tb.launch()

# Automatically open the TensorBoard URL in the default web browser
webbrowser.open(url)

# Print the URL where TensorBoard is accessible
print(f"TensorBoard is running at {url}")
