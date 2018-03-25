# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

KFOLDS = 10
KNEIGHBORS = 5

with open("diabetes.csv", 'r') as csvfile:
    data = pd.read_csv(csvfile)

numeric_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data_numeric = data[numeric_columns]
data_normalized = (data_numeric - data_numeric.min()) / (data_numeric.max() - data_numeric.min())

def knn(trainingFold, testFold, numNeighbors):
    testFoldCopy = pd.DataFrame(testFold.values, columns=testFold.columns)
    for index, instance in testFold.iterrows():
        newTrainingDataset = trainingFold
        newTrainingDataset['Distance'] = newTrainingDataset.sub(instance).pow(2).sum(1).pow(0.5)
        newTrainingDataset = newTrainingDataset.sort_values('Distance')
        kNeighbors = newTrainingDataset[0:numNeighbors]
        #print kNeighbors

        if (kNeighbors[kNeighbors.Outcome == 0]['Outcome'].count() > kNeighbors[kNeighbors.Outcome == 1]['Outcome'].count()):
            testFoldCopy.at[(index, 'Outcome')] = 0
        else:
            testFoldCopy.at[(index, 'Outcome')] = 1

    return testFoldCopy

def fold_i_of_k(dataset, i, k):
    n = len(dataset)
    return dataset[n*(i-1)//k:n*i//k]

def calculateAccuracyAndF1(testFold, results):
    incorrect = 0
    fp = fn = vp = vn = 0
    for i in range(len(testFold)):
        if testFold.at[i, 'Outcome'] != results.at[i, 'Outcome']:
            incorrect += 1 # For acurracy
            if results.at[i, 'Outcome'] == 0:
                fn += 1
            else:
                fp += 1
        else:
            if results.at[i, 'Outcome'] == 0:
                vn += 1
            else:
                vp += 1

    rev =  vp / float(vp + fn)
    prec = vp / float(vp + fp)

    f1 = 2 * (prec * rev / float(prec + rev))

    return (1 - (incorrect/float(len(testFold)))), f1

def main():
  shuffle_data = data_normalized.sample(frac=1)
  diabetes_true = shuffle_data[shuffle_data.Outcome == 1]
  diabetes_false = shuffle_data[shuffle_data.Outcome == 0]

  folds_false = [fold_i_of_k(diabetes_false, i+1, KFOLDS) for i in range(KFOLDS)]
  folds_true = [fold_i_of_k(diabetes_true, i+1, KFOLDS) for i in range(KFOLDS)]

  folds = [pd.DataFrame(np.concatenate((folds_true[i], folds_false[i]), axis=0), columns=shuffle_data.columns) for i in range(KFOLDS)]

  accuracy = [0 for i in range(KFOLDS)]
  f1_score = [0 for i in range(KFOLDS)]

  print("KFOLDS = %d \t KNEIGHBORS = %d" % (KFOLDS, KNEIGHBORS))

  for i in range(KFOLDS):
      # Concatenate and remove outcome
      testing_fold_np = np.concatenate((folds_true[i], folds_false[i]), axis=0)
      original_testing_fold = pd.DataFrame(testing_fold_np, columns=shuffle_data.columns)
      testing_fold_np = np.delete(testing_fold_np, 8, axis=1)
      testing_fold = pd.DataFrame(testing_fold_np, columns=shuffle_data.columns[0:8]) #without Outcome

      training_fold_np = None
      for j in range(KFOLDS):
          if (i != j):
              if training_fold_np is None:
                  training_fold_np = np.concatenate((folds_true[j], folds_false[j]), axis=0)
              else:
                  training_fold_np = np.concatenate((training_fold_np, folds_true[j], folds_false[j]), axis=0)
      training_fold = pd.DataFrame(training_fold_np, columns=shuffle_data.columns)

      #display(testing_fold)
      #display(training_fold)

      results = knn(training_fold, testing_fold, KNEIGHBORS);
      #print("RESULTADOS")
      #display(results)

      acc, f1 = calculateAccuracyAndF1(original_testing_fold, results)
      accuracy[i] = acc
      f1_score[i] = f1
      print "FOLD #%d ->  acc:%f  f1:%f" % (i, acc, f1)

  accuracy_avg = np.average(accuracy)
  accuracy_std = np.std(accuracy)
  f1_avg = np.average(f1_score)
  f1_std = np.std(f1_score)
  print "Acurácia   -> \tMédia: %.2f\tDesvio Padrão: %.2f" % (accuracy_avg, accuracy_std)
  print "Escore F-1 -> \tMédia: %.2f\tDesvio Padrão: %.2f" % (f1_avg, f1_std)

main()
