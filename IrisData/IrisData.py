import matplotlib.pyplot as plt
from random import randint
import numpy as np
import keras

dataNames = ["sepal length","sepal width","petal length","petal width","class"]
classNames ={"Iris-setosa":'k',"Iris-versicolor":'b',"Iris-virginica":'r'}
classNums ={"Iris-setosa":2,"Iris-versicolor":0,"Iris-virginica":1}

with open("iris.data.txt",'r') as f:
    data=f.readlines()
for i in range(len(data)):
    data[i] = data[i].strip().split(',')
flowers = []

for row in data:
    flower = {}
    for att in range(len(dataNames)):
        try:
            flower[dataNames[att]] = float(row[att])
        except:
            flower[dataNames[att]] = str(row[att])
            
    flowers.append(flower)

if(False):
    for i in range(4):
        for ii in range(4):
            if(ii>i):
                for flower in flowers:
                
                    x=(flower[dataNames[i]])
                    y=(flower[dataNames[ii]])
                    color = classNames[flower[dataNames[4]]]
                    plt.scatter(x,y,c=color)
                plt.xlabel(dataNames[i])
                plt.ylabel(dataNames[ii])
                plt.show()


#remove setosa
if(False):
    print(len(flowers))
    index = 0
    while(index < len(flowers)):
    
        if(classNames[flowers[index]["class"]]=="k"):
            flowers.pop(index)
            continue
        index+=1

print(len(flowers))
print(flowers[1])

#a=input()
#split data into test and training

testingIndex = []
while(len(testingIndex)<20):
    r = randint(0,len(flowers))
    if(r not in testingIndex):
        testingIndex.append(r)


trainingData =[]
classData = []
testData = []
testClass = []
for i in range(len(flowers)):
    if i not in testingIndex:
        for ii in range(len(dataNames)-1):
            trainingData.append(flowers[i][dataNames[ii]])
        classData.append(classNums[flowers[i][dataNames[-1]]])
    else:
        for ii in range(len(dataNames)-1):
            testData.append(flowers[i][dataNames[ii]])
        testClass.append(classNums[flowers[i][dataNames[-1]]])

#format data to correct dementions
testData = np.array(testData).reshape(int(len(testData)/(len(dataNames)-1)),len(dataNames)-1)
trainingData = np.array(trainingData).reshape(int(len(trainingData)/(len(dataNames)-1)),len(dataNames)-1)
testClass = np.array(testClass).reshape(len(testClass),1)
testClass = keras.utils.to_categorical(testClass,num_classes = 3)
classData=np.array(classData).reshape(len(classData),1)
classData = keras.utils.to_categorical(classData,num_classes = 3)

        
#make model

model = keras.models.Sequential()
model.add(keras.layers.Dense(4,input_shape=(4,),activation='relu'))
model.add(keras.layers.Dense(16,activation="sigmoid"))
model.add(keras.layers.Dense(16,activation="sigmoid"))
model.add(keras.layers.Dense(3,activation="sigmoid"))


#select training
model.compile(optimizer = 'rmsprop' ,loss = 'categorical_crossentropy',metrics = ['accuracy'])

#model.compile(optimizer = 'sgd' ,loss = 'mean_squared_error' ,metrics = ['accuracy'])

#train in three sessions
hist = model.fit(trainingData,classData,epochs = 500, batch_size = 10)
hist2 = model.fit(trainingData,classData,epochs = 250, batch_size = 32)
hist3 = model.fit(trainingData,classData,epochs = 250, batch_size = 64)

#plot loss
plt.plot(np.arange(0,500),hist.history["loss"])
plt.plot(np.arange(0,250),hist2.history["loss"])
plt.plot(np.arange(250,500),hist3.history["loss"])
plt.show()

#run testdata
pred = model.predict(testData)

#check results
totalcorrect  = 0;
for j in range(len(pred)):
    test = pred[j]
    guess = 0;
    correct = 0;
    for i in range(len(test)):
        if (test[guess]<test[i]):
            guess = i
        if (testClass[j][correct]<testClass[j][i]):
            correct = i
    
    if(guess == correct):
        totalcorrect+=1

print(totalcorrect/20*100)
#print(pred)
#print(testClass)



