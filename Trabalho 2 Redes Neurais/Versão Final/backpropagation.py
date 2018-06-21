# -*- coding: utf-8 -*-

# Usage: python3 backpropagation.py network.txt initial_weights.txt dataset.txt

# Authors: Augusto Bennemann and Fabricio Mazzola

import t2
import sys

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("Usage: python3 backpropagation.py network.txt initial_weights.txt dataset.txt")
  else:
    t2.setBackPropagationCheck(True)
    network = t2.createNetwork(sys.argv[1], sys.argv[2], 0.05)
    trainingSet = t2.createTrainSet(sys.argv[3])

    #print(network)
    #print(trainingSet)
    #t2.setDebug(True)


    J = network.calculateNetworkError2(trainingSet.instances)
    #print("\tDataset completo processado. Calculando gradientes regularizados")
    network.backPropagation2(trainingSet.instances, J)





