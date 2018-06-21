# -*- coding: utf-8 -*-

# Usage: python3 t2.py datasetname
#   where datasetname is part of {wine, cancer, diabetes, ionosphere}

# Authors: Augusto Bennemann and Fabricio Mazzola

import re
import math
import numpy as np
import csv
import random
import operator
import copy
import sys

"""
# Display file input on Google Colab
try:
  from google.colab import files
  uploaded = files.upload()
except ImportError:
  pass
"""

DEBUG = False
KFOLDS = 10
original_dataset = None
preditiveAttributes = {}
allAttributes = {}
EXPECTED_COLUMN = ""
dataset = None
np_original_dataset = None
possibleClasses = {}

# PRINT HELPERS

def log(s):
  if DEBUG:
    print(s)

def setDebug(value):
  global DEBUG
  DEBUG = value

def printFloatPrecision5(l):
  return "  ".join("%.5f" % a for a in l)

def twoDimensionPrintFloatPrecision5(l, preffix):
  msg = preffix
  for l2 in l:
    msg += " ".join("%.5f" % a for a in l2)
    msg += "\n"
    msg += preffix
  return msg

# DATASET READING AND PROCESSING
def openFile(filename, delimiter):
  """
  ATENCAO: 
  (i) a classe a ser prevista deve ser a ultima coluna do dataset
  (ii) a primeira linha deve conter o nome de cada atributo
  """
  with open(filename, 'r') as file:
    global preditiveAttributes, allAttributes, original_dataset, EXPECTED_COLUMN
    lines = csv.reader(file, delimiter=delimiter)
    list_lines = list(lines)

    preditiveAttributes = {x:i for i,x in enumerate(list_lines[0][:-1])}
    allAttributes = {x:i for i,x in enumerate(list_lines[0])} #Includes Class (last column)

    original_dataset = list_lines[1:]
    #EXPECTED_COLUMN = -1 #list_lines[0][-1]
    
def countClasses(dataset, attribute_index=EXPECTED_COLUMN):
  frequency = {} # frequency of a value
  
  for line in dataset:
    if line[attribute_index] in frequency:
      frequency[line[attribute_index]] += 1.0
    else:
      frequency[line[attribute_index]] = 1.0
      
  return frequency
      
def generatePossibleClasses(original_dataset):
  countclasses = countClasses(original_dataset, -1)
  possibleClasses = {}
  for possibleClass in countclasses:
    if not possibleClass in possibleClasses:
      possibleClasses[possibleClass] = len(possibleClasses)
  return possibleClasses

def normalize_data(dataset, high=1.0, low=0.0):
    mins = np.min(dataset, axis=0)
    maxs = np.max(dataset, axis=0)
    
    rng = maxs - mins
    for i,d in enumerate(rng):
      if d == 0:
        rng[i] = 0.0000001 # avoid division by zero
      
    return high - (((high - low) * (maxs - dataset)) / rng)


# NEURAL NETWORK MODEL
class Neuron:
  delta = 0
  activation = 0
  weights = []
  
  def __init__(self, initialWeights):
    self.weights = initialWeights
    self.delta = 0
    self.activation = 0
 
  def setDelta(self, delta):
    self.delta = delta
    
  def getDelta(self):
    return self.delta
  
  def __str__(self):
    return "  ".join("%.5f" % w for w in self.weights)
    
class Layer:
  neurons = []
  
  def __init__(self, weights):
    self.neurons = []
    for w in weights:
      self.neurons.append(Neuron(w))
    
  def __str__(self):    
    msg = ""
    for i, n in enumerate(self.neurons):
      msg += "\t%s\n" % n
    return msg
  
class Network:
  regularizationFactor = 0
  alpha = 0
  layers = []
  
  def __init__(self, regularizationFactor, layersWeights, inputSize, alpha=0.0001):
    print("Inicializando rede com a seguinte estrutura de neuronios por camadas: [%d %s]" % (inputSize, " ".join("%d" % len(a) for a in layersWeights)))
    
    self.regularizationFactor = regularizationFactor
    self.alpha = alpha
    self.layers = []
    for lw in layersWeights:
      self.layers.append(Layer(lw))
    
  def __str__(self):
    msg = "Parametro de regularizacao lambda=%.3f\n" % self.regularizationFactor
    for i, l in enumerate(self.layers):
      msg += ("\nTheta%d inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):\n" % (i+1))
      msg += str(l)
    return msg
  
  def activationFunction(self, x):
    return (1 / (1 + math.exp(-x)))
  
  def derivativeFunction(self, output):
    return output*(1.0-output)
    
  def updateWeights(self, gradients, J):
    log("\n ---> Atualizando os pesos <--- \n")
    for l, layer in enumerate(self.layers):
      for n, neuron in enumerate(layer.neurons):
        neuron.weights = list(np.array(neuron.weights) - self.alpha * J * np.array(gradients[l][n]))

  def calculateNetworkError2(self, instances):
    log("Calculando erro/custo J deste mini-batch")
    J = 0

    for i, d in enumerate(instances):
      log("\tProcessando exemplo de treinamento %d do mini-batch" % (i+1))

      inputPrecision5 = printFloatPrecision5(d['attributes'])
      expectedPrecision5 = printFloatPrecision5(d['expected'])

      log("\tPropagando entrada [%s]" % inputPrecision5)

      input = np.array([1] + d['attributes']) # [1] is for bias

      all_activations = []

      for k, layer in enumerate(self.layers):
        layer_activations =[]
        prevInput = np.array([1])
        results = []
        log("\t\ta%d: [%s]\n" % (k+1, printFloatPrecision5(input)))
        for neuron in layer.neurons:
          weights = np.array(neuron.weights)
          result = weights.T.dot(input)
          activation = self.activationFunction(result)

          layer_activations.append(activation)

          results.append(result)
          prevInput = np.append(prevInput, activation)        

        log("\t\tz%d: [%s]" % (k+2, printFloatPrecision5(results)))

        input = prevInput

        all_activations.append(layer_activations)

      d["results"] = np.round(input[1:], 5)

      d["activations"] = all_activations

      outputPrecision5 = printFloatPrecision5(input[1:]) # ignores first element (bias)
      log("\t\ta%d: [%s]\n" % (len(self.layers)+1, outputPrecision5)) # TODO meio gambiarra isso, mas senao não aparece o ultimo 'a'
      log("\t\tf(x): [%s]" % (outputPrecision5))

      log("\tSaida predita para o exemplo %d do mini-batch: [%s]" % (i+1, outputPrecision5))
      log("\tSaida esperada para o exemplo %d do mini-batch: [%s]" % (i+1, expectedPrecision5))

      Ji = sum([(-y * (math.log(f))) - ((1 - y)*(math.log(1 - f))) for y, f in zip(d["expected"], d["results"])])
      J += Ji

      log("\tJ do exemplo %d do mini-batch: %.3f\n" % (i+1, Ji))


    J = J / len(instances)

    S = 0

    for k, layer in enumerate(self.layers):
      for ni, nn in enumerate(self.layers[k].neurons):
        for i in range(1, len(nn.weights)):
          S += nn.weights[i] ** 2

    S = ( self.regularizationFactor / (2 * len(instances))) * S

    totalJ = J + S

    log("J total do mini-batch (com regularizacao): %.5f\n" % (totalJ))
    return totalJ
    
  def backPropagation2(self, instances, J):
    log("Rodando backpropagation")

    all_gradients = []  
    for i, d in enumerate(instances):
      log("\tCalculando gradientes com base no exemplo %d:" % (i+1))
      instance_gradients = []

      # Calculate delta for each neuron
      for j in reversed(range(len(self.layers))):
        deltas = []

        if j == len(self.layers)-1:
          # Calculate delta for last layer
          deltas = d['results'] - d['expected']
        else:
          # Calculate delta for hidden layers
          for ni, nn in enumerate(self.layers[j].neurons):
            partial_deltas = []
            for nj in self.layers[j+1].neurons: # for each neuron in next layer
              # ni+1 is used to ignore bias, since weights[0] is the weight related to bias
              delta_partial_result = nj.getDelta() * nj.weights[ni+1];
              partial_deltas.append(delta_partial_result)
            deltas.append(sum(partial_deltas)* self.derivativeFunction(d["activations"][j][ni]))

        for n in range(len(self.layers[j].neurons)):
          self.layers[j].neurons[n].setDelta(deltas[n]) # TODO faz sentido salvar aqui??? na real, isso ta sendo usado?

        log("\t\tdelta%d: [%s]" % (j+2, printFloatPrecision5(deltas)))

      #calcular gradiente dos thetas
      for j in reversed(range(-1, len(self.layers)-1)):
        weights = []
        delta = np.array([])
        activation = np.array([1])

        if j == -1: # first layer
          for n in range(len(self.layers[j+1].neurons)):
            weights.append([self.layers[j+1].neurons[n].weights])
            delta = np.append(delta, self.layers[j+1].neurons[n].delta)

          activation = np.append(activation, d['attributes']) 

        else:
          for n in range(len(self.layers[j].neurons)):
            activation = np.append(activation, d["activations"][j][n])

          for n in range(len(self.layers[j+1].neurons)):
            weights.append([self.layers[j+1].neurons[n].weights])
            delta = np.append(delta, self.layers[j+1].neurons[n].delta)


        activation = activation.reshape(activation.shape[0],-1)
        delta = delta.reshape(delta.shape[0],-1)
        weights = np.array(weights)

        gradient = delta.dot(activation.T)

        instance_gradients.append(gradient)

        log("\t\tGradientes de Theta%d com base no exemplo %d:" % (j+2, i+1))
        log(twoDimensionPrintFloatPrecision5(gradient, "\t\t\t"))

      all_gradients.append(instance_gradients)  
      
     
    log("\tDataset completo (ou mini-batch) processado. Calculando gradientes regularizados")

    finalD = []

    for ri, trow in enumerate(all_gradients[0]):
      weights_wo_bias = []
      for n in reversed(range(len(self.layers[len(self.layers)-ri-1].neurons))):
        weights_wo_bias.append([0] + self.layers[len(self.layers)-ri-1].neurons[n].weights[1:])

      P = np.array(list(reversed(weights_wo_bias))) * self.regularizationFactor

      D = np.zeros((trow.shape[0], trow.shape[1]))

      for rj in all_gradients:
        D += rj[ri]

      finalD.append((np.array(D) + P) / len(instances))

    for i in range(len(self.layers)):
      log("\t\tGradientes finais para Theta%d (com regularizacao):" % (i+1))
      log(twoDimensionPrintFloatPrecision5(finalD[len(finalD)-1-i], "\t\t\t"))

    self.updateWeights(list(reversed(finalD)), J) 
    
  def miniBatchRun(self, dataset, iterations, minibatchK):
    log("MINI BATCH RUN")
    print("RODANDO %d ITERAÇÕES:" % iterations),
    for i in range(iterations):
      if (i+25) % 25 == 0:
        print("%d ... " % i, end="")
      for mb in range(math.ceil(len(dataset.instances) / minibatchK)):
        currentMinibatchInstances = dataset.instances[mb*minibatchK:(mb+1)*minibatchK]
        J = self.calculateNetworkError2(currentMinibatchInstances)
        log("\tDataset completo do mini-batch #%d processado. Calculando gradientes regularizados" % mb)
        self.backPropagation2(currentMinibatchInstances, J)
      if (i+25) % 25 == 0:
        print("J = %f " % J)
        
  def verifyGradients(self, epsilon):
    print("Rodando verificacao numerica de gradientes (epsilon = %f)" % epsilon)
    for i in range(len(self.layers)):
      print("\tGradiente numerico de Theta%d: TODO" % (i+1))

  def verifyGradientsCorretude(self):
    print("Verificando corretude dos gradientes com base nos gradientes numericos:")
    for i in range(len(self.layers)):
      print("\tErro entre gradiente via backprop e gradiente numerico para Theta%d: TODO" % (i+1))
      
  def classifyInstances(self, testSet):
    results_all = []
    for i, instance in enumerate(testSet):

      inputPrecision5 = printFloatPrecision5(instance['attributes'])
      expectedPrecision5 = printFloatPrecision5(instance['expected'])

      log("\tPropagando entrada de teste [%s]" % inputPrecision5)

      input = np.array([1] + instance['attributes']) # [1] is for bias

      all_activations = []

      for k, layer in enumerate(self.layers):
        layer_activations =[]
        prevInput = np.array([1])
        results = []
        for neuron in layer.neurons:
          weights = np.array(neuron.weights)
          result = weights.T.dot(input)
          activation = self.activationFunction(result)
          layer_activations.append(activation)
          results.append(result)
          prevInput = np.append(prevInput, activation)        

        input = prevInput

        all_activations.append(layer_activations)

      instance["results"] = np.round(input[1:], 5)
      instance["activations"] = all_activations

      outputPrecision5 = printFloatPrecision5(input[1:]) # ignores first element (bias)
      log("\tSaida predita para o teste %d: [%s]" % (i+1, outputPrecision5))
      log("\tSaida esperada para o teste %d: [%s]\n" % (i+1, expectedPrecision5))
      
      results_all.append(list(np.round(input[1:], 5)))
    return results_all

class TrainOrTestSet:
  instances = []

  def __init__(self):
    self.instances = []

  def append(self, instance):
    self.instances.append(instance)
    
  def __str__(self):
    msg = "Conjunto de treinamento\n"
    for i, instance in enumerate(self.instances):
      msg += "\tExemplo %d\n\t\tx: [%s]\n\t\ty: [%s]\n" % (i+1, printFloatPrecision5(instance["attributes"]), printFloatPrecision5(instance["expected"]))
    return msg

def createNetwork(networkFilename, initialWeightsFilename, alpha):
  regularizationFactor = 0
  neuronsPerLayer = []
  initialWeights = []
  inputSize = 0

  with open(networkFilename) as f:
    line = f.readline()
    cnt = 1
    while line:

      if cnt == 1:
        regularizationFactor = float(line)
      else:
        if cnt == 2:
          inputSize = int(line)
        neuronsPerLayer.append(int(line))

      line = f.readline()
      cnt += 1

  with open(initialWeightsFilename) as f:
    line = f.readline()
    cnt = 1
    while line:

      splitLine = re.split(r';', line)

      layerInitialWeights = []
      for i in splitLine:
        layerInitialWeights.append([float(w) for w in re.findall(r'\d+\.\d+', i)])

      initialWeights.append(layerInitialWeights)

      line = f.readline()
      cnt += 1
  
  network = Network(regularizationFactor, initialWeights, inputSize, alpha) 
  
  return network

def createTrainSet(datasetFilename):
  trainSet = TrainOrTestSet()
  with open(datasetFilename) as f:
    
    line = f.readline()
    cnt = 1
    
    while line:
      instance = {}
      
      splitLine = re.split(r';', line)
      
      instance = {}
      instance['attributes'] = [float(a) for a in re.findall(r'\d+\.\d+', splitLine[0])]
      instance['expected'] = [float(a) for a in re.findall(r'\d+\.\d+', splitLine[1])]
      
      trainSet.append(instance)
      
      line = f.readline()
      cnt += 1
      
  return trainSet

def randomWeight():
  value = 0.0
  while value == 0.0:
    value = np.random.normal(0.0, 0.15) # values close to 0.0
  return value

def generateRandomWeights(neuronsPerLayer):
  weights = []
  
  for i, layer in enumerate(neuronsPerLayer):
    if i >= 1:
      weights.append([[randomWeight() for j in range(neuronsPerLayer[i-1] + 1)] for k in range(neuronsPerLayer[i])])
  return weights

def runPredefinedDataset(datasetName):
  """
  dataset: {diabetes, ionosphere, cancer, wine}
  """
  
  global dataset, np_original_dataset, original_dataset, EXPECTED_COLUMN, possibleClasses
    
  if datasetName == "diabetes":
    openFile("diabetes.csv", ',')
        
    for d in original_dataset:
      if d[-1] == '0':
        d[-1] = 0.0
      else: # 1
        d[-1] = 1.0
    
    np_original_dataset = np.array(original_dataset).astype(np.float)
    dataset = normalize_data(np.array(np_original_dataset).astype(np.float))
    possibleClasses = generatePossibleClasses(original_dataset)
    
    numberInputs = len(dataset[1]) - 1
    
    #generateFoldsAndTest(numberInputs, [numberInputs, 2, 1], 0.05, 0.1, 20, 700)
    generateFoldsAndTest(numberInputs, [numberInputs, 4, 1], 0.05, 0.1, 20, 700)
    #generateFoldsAndTest(numberInputs, [numberInputs, 8, 1], 0.05, 0.1, 20, 700)

  elif datasetName == "wine":
    openFile("wine_with_names.csv", ',')
    
    for d in original_dataset:
      if d[-1] == '1':
        d[-1] = 0.0
      elif d[-1] == '2':
        d[-1] = 0.5
      else: # 3
        d[-1] = 1.0
    
    np_original_dataset = np.array(original_dataset).astype(np.float)
    dataset = normalize_data(np.array(np_original_dataset).astype(np.float))
    possibleClasses = generatePossibleClasses(original_dataset)

    numberInputs = len(dataset[1]) - 1

    generateFoldsAndTest(numberInputs, [numberInputs, 5, 1], 0.05, 0.0, 20, 800)
    #generateFoldsAndTest(numberInputs, [numberInputs, 10, 1], 0.05, 0.0, 20, 800)
    #generateFoldsAndTest(numberInputs, [numberInputs, 15, 1], 0.05, 0.0, 20, 800)
    #generateFoldsAndTest(numberInputs, [numberInputs, 4, 2, 1], 0.05, 0.0, 20, 800)

  elif datasetName == "ionosphere":
    
    openFile("ionosphere_with_names.csv", ',')
    
    for d in original_dataset:
      if d[-1] == 'g':
        d[-1] = 0.0
      else: # 'b'
        d[-1] = 1.0
    np_original_dataset = np.array(original_dataset).astype(np.float)
    dataset = normalize_data(np.array(np_original_dataset).astype(np.float))
    possibleClasses = generatePossibleClasses(original_dataset)
    
    numberInputs = len(dataset[1]) - 1
    
    generateFoldsAndTest(numberInputs, [numberInputs, 4, 1], 0.05, 0.1, 20, 700) 
    #generateFoldsAndTest(numberInputs, [numberInputs, 6, 1], 0.05, 0.1, 20, 700) 
    #generateFoldsAndTest(numberInputs, [numberInputs, 8, 1], 0.05, 0.1, 20, 700) 
    #generateFoldsAndTest(numberInputs, [numberInputs, 4, 2, 1], 0.05, 0.1, 20, 700) 
    
    
  elif datasetName == "cancer":
    
    openFile("wdbc_with_names.csv", ',')
    
    for d in original_dataset:
      if d[-1] == 'B':
        d[-1] = 0.0
      else: # 'M'
        d[-1] = 1.0
    np_original_dataset = np.array(original_dataset).astype(np.float)
    dataset = normalize_data(np.array(np_original_dataset).astype(np.float))
    possibleClasses = generatePossibleClasses(original_dataset)
    
    numberInputs = len(dataset[1]) - 1
  
    generateFoldsAndTest(numberInputs, [numberInputs, 5, 1], 0.05, 0.1, 20, 700)
    #generateFoldsAndTest(numberInputs, [numberInputs, 10, 1], 0.05, 0.1, 20, 700)
    #generateFoldsAndTest(numberInputs, [numberInputs, 15, 1], 0.05, 0.1, 20, 700)
    
  
  else:
    print("Invalid dataset. Available options are: diabetes; wine; ionosphere; cancer.")
    return

def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return dataset[n*(i-1)//k:n*i//k]

def calculateAccuracyAndF1(testFold, results, possibleClassesLen):    
  if possibleClassesLen == 2:
    incorrect = 0
    fp = fn = vp = vn = 0
    range_size = 0.5

    for i in range(len(testFold)):
      if math.fabs(testFold[i][-1] - results[i][0]) >= range_size:
          incorrect += 1 # For acurracy
          if results[i][0] < range_size:
              fn += 1
          elif results[i][0] >= range_size:
              fp += 1
      else:
          if results[i][0] < range_size:
              vn += 1
          elif results[i][0] >= range_size:
              vp += 1

    print ("vp: " + str(vp) + "  vn: " + str(vn) + " fp: " + str(fp) + " fn: " + str(fn))

    rev =  vp / float(vp + fn)
    prec = vp / float(vp + fp)

    f1 = 2 * (prec * rev / float(prec + rev))
    
    return (1 - (incorrect/float(len(testFold)))), f1
    
  elif possibleClassesLen == 3:
    range_size = 1/3.0

    incorrect = 0
    revs = []
    precs = []
    f1s_sum = 0

    for j in range(3):
      fp = fn = vp = vn = 0
      for i in range(len(testFold)):
        result = results[i][0]

        if j*range_size < result and result <= (j+1)*range_size:
          if j*range_size <= testFold[i][-1] and testFold[i][-1] <= (j+1)*range_size:
            vp += 1
          else:
            fp += 1
            incorrect += 1 # For acurracy

        else:
          if j*range_size <= testFold[i][-1] and testFold[i][-1] <= (j+1)*range_size:
            fn += 1
            incorrect += 1 # For acurracy
          else:
            vn += 1


      print ("vp: " + str(vp) + "  vn: " + str(vn) + " fp: " + str(fp) + " fn: " + str(fn))

      rev = 0
      if vp + fn != 0:
        rev =  vp / float(vp + fn)

      prec = 0
      if vp + fp != 0:
        prec = vp / float(vp + fp)

      if (prec == 0 or rev == 0):
        f1 = 0
      else:
        f1 = (2 * (prec * rev / float(prec + rev)) )
      f1s_sum += f1

    return (1 - (incorrect/float(len(testFold)))/3), f1s_sum/3

def generateFoldsAndTest(numberInputs, networkFormat, alpha, regularizationFactor, minibatchK, iterations):
  
  dataset_copy = copy.copy(dataset)
  np.random.shuffle(dataset_copy)
  
  outcomes = { d:[] for i,d in enumerate(possibleClasses)}
  
  for i in dataset_copy:
    outcomes[((i[-1]))].append(i)
  
  folds_by_class = {}
  for index, d in enumerate(possibleClasses):
    folds_by_class[d] = [fold_i_of_k(outcomes[d], i+1, KFOLDS) for i in range(KFOLDS)]
  
  folds = [np.concatenate(tuple(folds_by_class[d][i] for d in possibleClasses), axis=0) for i in range(KFOLDS)]
  
  accuracy = [0 for i in range(KFOLDS)]
  f1_score = [0 for i in range(KFOLDS)]

  print("KFOLDS = %d \t" % KFOLDS)
  
  for i in range(KFOLDS):
      minlen = len(dataset)
      for v1 in folds_by_class.keys():
        for v2 in folds_by_class[v1]:
          minlen = min(minlen, len(v2))

      # Concatenate
      original_testing_fold = np.concatenate(tuple(folds_by_class[d][i][0:minlen-1] for d in possibleClasses), axis=0)

      training_fold = None
      for j in range(KFOLDS):
          if (i != j):
              if training_fold is None:
                  training_fold = np.concatenate(tuple(folds_by_class[d][j][0:minlen-1] for d in possibleClasses), axis=0)
              else:
                  newarray = np.concatenate(tuple(folds_by_class[d][j][0:minlen-1] for d in possibleClasses), axis=0)
                  training_fold = np.concatenate((training_fold, newarray), axis=0)

      # Create network
      initialWeights = generateRandomWeights(networkFormat)
    
      network = Network(regularizationFactor, initialWeights, numberInputs, alpha)
      log(network)

      # Train
      trainingSet = TrainOrTestSet()
      for instance in training_fold:
        trainingSet.append({"attributes": instance[:-1].tolist(), "expected": [instance[-1].tolist()]})

      network.miniBatchRun(trainingSet, iterations, minibatchK)

      fold_results = []
      testingSet = TrainOrTestSet()
      for instance in original_testing_fold:
        testingSet.append({"attributes": instance[:-1].tolist(), "expected": [instance[-1].tolist()]})

      fold_results = network.classifyInstances(testingSet.instances)
      #print("EXPECTED:")
      #print([x["expected"] for x in testingSet.instances])
      #print("FOLD RESULTS")
      #print(fold_results)
                     
      if len(possibleClasses) <= 3:
        acc, f1 = calculateAccuracyAndF1(original_testing_fold, fold_results, len(possibleClasses))
        accuracy[i] = acc
        f1_score[i] = f1
        print("FOLD #%d ->  acc:%f  f1:%f" % (i+1, acc, f1))

  if len(possibleClasses) <= 3:
    accuracy_avg = np.average(accuracy)
    accuracy_std = np.std(accuracy)
    f1_avg = np.average(f1_score)
    f1_std = np.std(f1_score)
    print("Acurácia   -> \tMédia: %.2f\tDesvio Padrão: %.2f" % (accuracy_avg, accuracy_std))
    print("Escore F-1 -> \tMédia: %.2f\tDesvio Padrão: %.2f" % (f1_avg, f1_std))
    return [f1_avg, f1_std]

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python3 t2.py datasetname")
  else:
    runPredefinedDataset(sys.argv[1])





