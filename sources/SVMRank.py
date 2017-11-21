import pandas as pd
import itertools
import numpy as np
from sklearn import svm

class SVMRank(object):
    '''A class to do SVM Ranking on a given dataframe of data points 

    Args:
        df (:obj:'dataframe' of :obj:'float'): includes ranking features and one query feature. Assume fips_code is always the first column and query feature is the second. {fips_code, <query>, <rank1>, <rank2> ...}
        k (int): the number of data points to be trained
        query_feature_i (int): index of the query_feature
        
    '''
    def __init__(self, df, k, query_feature_i=1):
        self.df = df
        self.k = k
        self.query_feature_i = query_feature_i

    '''Function to train svm by using top k data points in terms of the query feature

    Args:
        asc (bool): True if sorting query feature in ascending order. Meaning the lower the query feature value, the higher the input ranking
        k (int, optional): pick the top k data points to train svm

    Returns:
        weights of features: array
        
    '''
    def train_top_k(self, asc, k=-1):
        if k == -1:
            k = self.k
        sorted_df = self.sort_dataframe(asc=asc, column=self.query_feature_i)
        top_k_ds = sorted_df[0:k]
        x_train, y_train = self.transform_pair(top_k_ds)
        #print "x_train: ", x_train
        #print "y_train: ", y_train
        self.svmr = SVMR().fit(x_train, y_train)
        #print "weights: ", self.svmr.coef_.ravel()
        return self.svmr.coef_.ravel()

    '''Function to get final ranking of all n data points 

    Args:

    Returns:
        sorted rank_df: dataframe, {'fips_code', <query feature>, 'SVMRank'}
        
    '''
    def get_full_rank(self):
        if hasattr(self, 'svmr'):
            x = self.df.iloc[:, self.query_feature_i+1:]
            ranks = self.svmr.predict(x)
            cols = [self.df.columns[0]]
            cols.append(self.df.columns[1]) 
            cols.append('SVMRank')
            rank_df = pd.DataFrame(0.0, index=self.df.index, columns=cols)
            rank_df.iloc[:, :2] = self.df.iloc[:, :2]
            rank_df['SVMRank'] = ranks
            #print "ranks: ", ranks
            return rank_df.sort_values('SVMRank', ascending=False)
        else:
            raise ValueError("You must train the model before predicting!")

    '''
    Not a suitable test. Ground truth y values are purely query feature values. Just for reference.
    '''
    def score_test(self):
        if hasattr(self, 'svmr'):
            x = self.df.iloc[:, self.query_feature_i+1:]
            true_y = np.argsort(self.df.iloc[:, self.query_feature_i])
            score = self.svmr.score(x, true_y)
            #print "score: ", score
            return score
        else:
            raise ValueError("You must train the model before predicting!")
        

        
    '''Function to transform n-class data points to 2-class training data for svmrank

    Args:
        df (:obj:'dataframe' of :obj:'float'): has k data points

    Returns:
        res_x: ndarray, (k, number of ranking features)
        res_y: array, (k, )
        
    '''
    def transform_pair(self, df):
        combinations = itertools.combinations(range(df.shape[0]), 2)
        y = df.iloc[:, self.query_feature_i]
        x = df.iloc[:, self.query_feature_i+1:]
#        print "ranking features: ", x
#        print "query features: ", y
        res_x = []
        res_y = []
        for k, (i, j) in enumerate(combinations):
            if y[i] == y[j]:
                continue
            res_x.append(x.ix[i]-x.ix[j]) 
            res_y.append(np.sign(y[i]-y[j]))
            if res_y[-1] != (-1)**k:
                res_y[-1] = -res_y[-1]
                res_x[-1] = -res_x[-1]
        return np.asarray(res_x), np.asarray(res_y).ravel()
    
    '''Function to sort the instance dataframe given a column    

    Args:
        asc (bool): True if in ascending order
        column (int, optional): sorting based on this column

    Returns:
        dataframe of numbers: sorted based on given column name in ascending order or descending
        
    '''
    def sort_dataframe(self, asc, column):
        return self.df.sort_values(self.df.columns[column], ascending=asc)



class SVMR(svm.LinearSVC):
    def fit(self, x, y):
        super(SVMR, self).fit(x, y)
        return self

    def predict(self, x):
        if hasattr(self, 'coef_'):
            return np.dot(x, self.coef_.T).ravel()
        else:
            raise ValueError("You must train the model before predicting!")

    def score(self, x, y):
        return np.mean(np.argsort(super(SVMR, self).predict(x)) == y)
