# Loading libararies
from random import choice
from random import uniform
from numpy.random import randint
from numpy import random
from operator import attrgetter
import os
import numpy as np
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPool2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam, SGD ,RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import models
from numba import jit, cuda



@jit(target_backend='cuda')
#Loading Data and Visualization
def plot_high_quality_bar_chart(classes, data, dataset_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, data, color=['skyblue', 'orange'])

   
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 1),
                ha='center', va='bottom', color='black', fontsize=12)

    ax.set(title=f"Dataset's Distribution for Each Class ({dataset_type})",
           xlabel="Classes",
           ylabel="# Images")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f'dataset_distribution_{dataset_type.lower()}_high_quality.png', dpi=800, bbox_inches='tight')

    plt.show()
def data_visualization(classes, data, dataset_type):
    plot_high_quality_bar_chart(classes, data, dataset_type)
def load_data(data_path):
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    train_datagen = ImageDataGenerator(
        validation_split=0.1,   
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

   
    train_generator = train_datagen.flow_from_directory(
        os.path.join(train_dir),
        target_size=(48, 48),
        batch_size=8,
        class_mode='categorical',
        classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise' ,'neutral'],
        shuffle=True,
        subset='training')  

  
    validation_generator = train_datagen.flow_from_directory(
        os.path.join(train_dir),
        target_size=(48, 48),
        batch_size=8,
        class_mode='categorical',
        classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral'],
        shuffle=True,
        subset='validation')  

    test_generator = test_datagen.flow_from_directory(
        os.path.join(test_dir),
        target_size=(48, 48),
        batch_size=8,
        class_mode='categorical',
        classes=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral'],
        shuffle=False)

    

    return train_generator, test_generator, validation_generator

#Enter path
data_path_fer = "Enter your path"
train_generator, test_generator, validation_generator = load_data(data_path_fer)


y_train = to_categorical(train_generator.classes, num_classes=7)
y_validation = to_categorical(validation_generator.classes, num_classes=7)  
y_test = to_categorical(test_generator.classes, num_classes=7)

print("Training label shape:", y_train.shape)
print("Training class indices:", train_generator.class_indices)
print("Training class labels:", train_generator.class_indices.keys())

print("Testing label shape:", y_test.shape)
print("Testing class indices:", test_generator.class_indices)
print("Testing class labels:", test_generator.class_indices.keys())

class sol:
    list = []
    fitness = 0.0
#Parameters setting
numPop = 50
numMutation = 26
numCrossover = 12
maximumGeneration=250
#Define Model
def CNN_model( f1, f2, k ,a1,a2,d1,d2,ep, input_shape=(48,48,3)):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3,3), padding='same', strides=(1, 1), name='conv1', activation='LeakyReLU',
                         kernel_initializer=glorot_uniform(seed=0)))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(f1, (k,k), padding='same', strides=(1, 1), name='conv2', activation=a1,
                         kernel_initializer=glorot_uniform(seed=0)))
    model.add(Dropout(d1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(layers.Flatten())
    model.add(Dense(f2, activation=a2, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
    model.add(Dropout(d2))
    model.add(BatchNormalization())
    model.add(Dense(7, activation='softmax', kernel_initializer=glorot_uniform(seed=0), name='fc2'))
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(0.0001), metrics = ["accuracy"])
    es = EarlyStopping(monitor="val_accuracy", patience = 7)
    model.fit(train_generator, validation_data=(validation_generator), epochs=ep, batch_size = 8, callbacks = [es], verbose=0)
    return model
#Fitness function
def fitness_evaluation(Model):
    metrics = Model.evaluate(test_generator)
    return metrics[1]
#Solutions
def fitness(solution):
     f1=solution[0]
     f2=solution[1]
     k=solution[2]
     a1=solution[3]
     a2=solution[4]
     d1=solution[5]
     d2=solution[6]
     ep=solution[7]
     Model=CNN_model( f1, f2, k ,a1,a2,d1,d2,ep)
     return fitness_evaluation(Model)
#Generate Solutions
def generateSol():
    parameters = []
    f1 = choice([32, 64])
    parameters.append(f1)
    f2 = choice([256, 512])
    parameters.append(f2)
    k = choice([3, 5])
    parameters.append(k)
    a1 = choice(['ReLU', 'LeakyReLU', 'ELU'])
    parameters.append(a1)
    a2 = choice(['ReLU', 'LeakyReLU', 'ELU'])
    parameters.append(a2)
    d1 = round(uniform(0.1, 0.5), 1)
    parameters.append(d1)
    d2 = round(uniform(0.1, 0.5), 1)
    parameters.append(d2)
    ep = randint(100, 150)
    parameters.append(ep)
    return parameters
#Generate population
def generatePop(sizePop):
    listPop = []
    for i in range(sizePop):
        solution = generateSol()
        fit = fitness(solution)
        solTemp = sol()
        solTemp.list = solution
        solTemp.fitness = fit
        listPop.append(solTemp)
    return listPop
#Define mutation method
def mutation(solution):

    chromosome = []
    for a in solution.list:
        chromosome.append(a)
    list1 = []
    for i in range(len(chromosome)):
        list1.append(i)
    arr = np.array(list1)
    random.shuffle(arr)
    i = arr[0]
    j = arr[1]
    k = arr[2]
    if(i == 0):
        t = choice([32, 64])
        chromosome[i] = t
    if(j == 0):
        t = choice([32, 64])
        chromosome[j] = t
    if(k == 0):
        t = choice([32, 64])
        chromosome[k] = t
    if(i == 1):
        t = choice([256, 512])
        chromosome[i] = t
    if(j == 1):
        t = choice([256, 512])
        chromosome[j] = t
    if(k == 1):
        t = choice([256, 512])
        chromosome[k] = t
    if(i == 2):
        t = choice([3, 5])
        chromosome[i] = t
    if(j == 2):
        t = choice([3, 5])
        chromosome[j] = t
    if(k == 2):
        t = choice([3, 5])
        chromosome[k] = t
    if(i == 3):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[i] = t
    if(j == 3):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[j] = t
    if(k == 3):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[k] = t
    if(i == 4):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[i] = t
    if(j == 4):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[j] = t
    if(k == 4):
        t = choice(['ReLU', 'LeakyReLU', 'ELU'])
        chromosome[k] = t
    if(i == 5):
        b = random.uniform(-0.1, 0.1)
        chromosome[i]+=b
        chromosome[i]=min(chromosome[i],0.5)
        chromosome[i]=max(chromosome[i],0.1)
    if(j == 5):
        b = random.uniform(-0.1, 0.1)
        chromosome[j]+=b
        chromosome[j]=min(chromosome[j],0.5)
        chromosome[j]=max(chromosome[j],0.1)
    if(k == 5):
        b = random.uniform(-0.1, 0.1)
        chromosome[k]+=b
        chromosome[k]=min(chromosome[k],0.5)
        chromosome[k]=max(chromosome[k],0.1)
    if(i == 6):
        b = random.uniform(-0.1, 0.1)
        chromosome[i]+=b
        chromosome[i]=min(chromosome[i],0.5)
        chromosome[i]=max(chromosome[i],0.1)
    if(j == 6):
        b = random.uniform(-0.1, 0.1)
        chromosome[j]+=b
        chromosome[j]=min(chromosome[j],0.5)
        chromosome[j]=max(chromosome[j],0.1)
    if(k == 6):
        b = random.uniform(-0.1, 0.1)
        chromosome[k]+=b
        chromosome[k]=min(chromosome[k],0.5)
        chromosome[k]=max(chromosome[k],0.1)
    if(i == 7):
        b = round(random.uniform(-5, 5))
        chromosome[i]+=b
        chromosome[i]=min(chromosome[i],150)
        chromosome[i]=max(chromosome[i],100)
    if(j == 7):
        b = round(random.uniform(-5, 5))
        chromosome[j]+=b
        chromosome[j]=min(chromosome[j],150)
        chromosome[j]=max(chromosome[j],100)
    if(k == 7):
        b = round(random.uniform(-5, 5))
        chromosome[k]+=b
        chromosome[k]=min(chromosome[k],150)
        chromosome[k]=max(chromosome[k],100)
    return chromosome
#Define Crossover method
def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for i in range(len(parent1)):
            prob = random.uniform(0,1.)
            if prob>0.5:
                child1.append(parent2[i])
                child2.append(parent1[i])
            else:
                child1.append(parent1[i])
                child2.append(parent2[i])
    return [child1,child2]

def generateMutationPopulation(listAllpop):
    mutationList = []
    for i in range(numMutation):
        index = random.randint(0, numPop-1)
        solutionTemp = mutation(listAllpop[index])
        fitnessTemp = fitness(solutionTemp)
        objectSol = sol()
        objectSol.list = solutionTemp
        objectSol.fitness = fitnessTemp
        mutationList.append(objectSol)
    return mutationList

def concateParentsAndChild(solutionAndFitnessList, mutationList,listCrossover):
    for i in mutationList:
        solutionAndFitnessList.append(i)
    for j in listCrossover:
        solutionAndFitnessList.append(j)
    return solutionAndFitnessList

def sortList(solutionAndFitnessList):
    solutionAndFitnessList.sort(key=attrgetter('fitness'), reverse=True)
    return solutionAndFitnessList

def generateCrossoverPopulation(listAllPop):
    crossoverlist=[]
    for i in range(numCrossover):
        list1=[]
        for i in range(len(listAllPop)):
             list1.append(i)
        index = np.array(list1)
        random.shuffle(index)
        '''print(listAllPop[index[0]].list)
        print(listAllPop[index[1]].list)
        print('****************')'''
        solutionTemp=uniform_crossover(listAllPop[index[0]].list, listAllPop[index[1]].list)
        sol1=solutionTemp[0]
        sol2=solutionTemp[1]
        fitnessTemp = fitness(sol1)
        objectSol = sol()
        objectSol.list = sol1
        objectSol.fitness = fitnessTemp
        crossoverlist.append(objectSol)

        fitnessTemp1 = fitness(sol2)
        objectSol1 = sol()
        objectSol1.list = sol2
        objectSol1.fitness = fitnessTemp1
        crossoverlist.append(objectSol1)

    return crossoverlist

listAllPop = generatePop(numPop)
listMutation = generateMutationPopulation(listAllPop)
listCrossover = generateCrossoverPopulation(listAllPop)
for i in range(maximumGeneration):
  mutationList=generateMutationPopulation(listAllPop)
  listCrossover = generateCrossoverPopulation(listAllPop)
  solutionAndFitnessList=concateParentsAndChild(listAllPop,mutationList,listCrossover)
  solutionAndFitnessList=sortList(solutionAndFitnessList)
  listAllPop=solutionAndFitnessList[0:numPop]

  print('generation = ',i,'  ************************************  ')
  print('Best solution is = ',listAllPop[0].list )
  print('Best fitness is = ',listAllPop[0].fitness )

