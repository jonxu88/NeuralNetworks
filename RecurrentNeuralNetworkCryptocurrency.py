#Following sentdex: https://www.youtube.com/watch?v=ne-dpRdNReI


import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
#TensorBoard gives info about the model, ModelCheckpoint allows us to save weights
#couldn't get checkpoint to work!!
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint 


#Predict the value of a cryptocurrency 3 minutes into the future based on the last 60 minutes (data given once a minute)
#val_accuracy gets to 0.56 for LTC-USD, meaning that the model predicts buy/sell correctly 56% of the time

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


# If future price is greater than current price, we should buy!
def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

#Function to preprocess data, kind of lost here!
def preprocess_df(df):
    df = df.drop('future', 1) #drop the future column, otherwise the network will learn very quickly!
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change() #pct_change computes the percent (decimal) change between the current and previous element
            df.dropna(inplace=True) #If any entries of the dataframe are missing then remove that row
            df[col] = preprocessing.scale(df[col].values) #scale values to values from 0 to 1
        
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN) #deque is just a list that pops out old items when more items are added to it
    #maybe a better name would be "prev_mins"

    for i in df.values:
        prev_days.append([n for n in i[:-1]]) #removes the "target" column and append data to prev_days
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys=[]
    sells=[]

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

#The cryptocurrency data can be found at https://pythonprogramming.net/static/downloads/machine-learning-data/crypto_data.zip
#The LTC-USD file has lightcoin data, each of the names is a name of a column
#Make sure to run python from the folder containing the folder "crypto_data" so it reads the csv correctly

main_df = pd.DataFrame()

#Collect all the "close" and "volume" data from each .csv file, label them with the names of their respective
#cryptocurrencies, then put it all in one pandas dataframe called main_df
ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = f"crypto_data/{ratio}.csv"
    df = pd.read_csv(dataset, names=["time", "low","high","open","close","volume"])
    #print(df.head())
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
    df.set_index("time", inplace=True)
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)


#Add a new column which gives the future price of RATIO_TO_PREDICT
main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)

#Add a column named target, where if target is 1 then it means we should buy (since future price is higher),
#otherwise 0 means we shouldn't buy
#map(function,iterables) executes function for each item in the iterable
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))


#Start taking data out of the sample for validation purposes
#Sorted returns a sorted list of an iterable object using its inherent order, in our case, time has an ordering
times = sorted(main_df.index.values)

#Get the last 5 percent of the time values
last_5pct = times[-int(0.05*len(times))]

#Split data into validation and training data
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

#The below differs from Sentdex, probably because we have no GPU, we need to change these to np.arrays
train_x  = np.array(train_x)
train_y  = np.array(train_y)
validation_x = np.array(validation_x)
validation_y = np.array(validation_y)

#print(f"train data: {len(train_x)} validation: {len(validation_x)}")
#print(f"Do not buys: {train_y.count(0)}, buys: {train_y.count(1)}")
#print(f"VALIDATIONS... Do not buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

######
#Construct the neural network
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation='tanh',return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, activation='tanh'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
#couldn't get checkpoint to work!!
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard],
)

