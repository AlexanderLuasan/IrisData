import matplotlib.pyplot as plt


dataNames = ["sepal length","sepal width","petal length","petal width","class"]
classNames ={"Iris-setosa":'k',"Iris-versicolor":'b',"Iris-virginica":'r'}

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