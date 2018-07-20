from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
Dense(32, input_dim=784),
Activation('relu'),
Dense(10),
Activation('softmax'),
])
#or this way
#model = Sequential()
#model.add(Dense(32, input_dim=784))
#model.add(Activation('relu'))


#equal expression of model 
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, batch_input_shape=(None, 784)))
# note that batch dimension is "None" here,
# so the model will be able to process batches of any size.</pre>

model = Sequential()
model.add(Dense(32, input_dim=784))


#equal expression of model 
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))

model = Sequential()
model.add(LSTM(32, batch_input_shape=(None, 10, 64)))

model = Sequential()
model.add(LSTM(32, input_length=10, input_dim=64))

#Merge
from keras.layers import Merge

left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))

merged = Merge([left_branch, right_branch], mode='concat') #张量串联

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(10, activation='softmax'))