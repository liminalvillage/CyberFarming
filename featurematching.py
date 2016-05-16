# multivariate lineair regression on the plantlet features

import numpy as np
import pandas as pd
import csv















if __name__ == '__main__':
    path = './output/saliency3/extfeat2/features_plant0.csv'
    features = pd.DataFrame(pd.read_csv(path, sep=','))

    correlations = features.corr()
    print correlations
    corCols = correlations.columns
    averages = []
    index = len(features.columns)
    for co in corCols:
        index = index -1
        cor = correlations[co]
        # average pearson correlations, highest average has higest over all correlations
        avrg = np.mean(cor)
        averages.append([avrg, index])
    newavrg = sorted(averages, reverse=True)
    print newavrg
    print newavrg[:5]

    path = './output/saliency3/extfeat2/features_plant2.csv'
    features = pd.DataFrame(pd.read_csv(path, sep=','))

    correlations = features.corr()
    print correlations
    corCols = correlations.columns
    averages = []
    index = len(features.columns)
    for co in corCols:
        index = index -1
        cor = correlations[co]
        # average pearson correlations, highest average has higest over all correlations
        avrg = np.mean(cor)
        averages.append([avrg, index])
    newavrg = sorted(averages, reverse=True)
    print newavrg
    print newavrg[:5]





    #print features.corr()
    #amountCol = len(features.columns)
    #i = 0
    #j = 1
    #while i < amountCol:
    #    while j<amountCol:
    #        col = features.iloc[:,i:j]
    #        j = j + 1
    #        i = i + 1
    #        print col


    #col = features.iloc[:,0:1]
    #print col
    #rest = features.iloc[:,1:]
    #print rest
