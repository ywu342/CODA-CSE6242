import pandas as pd
import itertools
import numpy as np
from sklearn import svm
import data_utils
import random

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

    '''Function to train svm by using randomly selected data points

    Args:
        k (int, optional): pick the random k data points to train svm

    Returns:
        weights of features: array
        
    '''
    def train_random_k(self, k=-1):
        if k == -1:
            k = self.k
        rand_inds = random.sample(range(0, self.df.shape[0]), k)
        #print "rand_inds: ", rand_inds
        rand_samples = self.df.iloc[rand_inds]
        #print "rand_samples: ", rand_samples
        x_train, y_train = self.transform_pair(rand_samples)
        #print "x_train: ", x_train
        #print "y_train: ", y_train
        self.svmr = SVMR().fit(x_train, y_train)
        #print "weights: ", self.svmr.coef_.ravel()
        return self.svmr.coef_.ravel()

    '''Function to train svm by using multiple iterations of randomly selected data points

    Args:
        k (int, optional): pick the random k data points 
        iterations (int): number of iterations

    Returns:
        weights of features: array
        
    '''
    def train_random_k_iter(self, iterations, k=-1):
        if k == -1:
            k = self.k
        x_train_all = []
        y_train_all = []
        for i in xrange(iterations):
            rand_inds = random.sample(range(0, self.df.shape[0]), k)
            rand_samples = self.df.iloc[rand_inds]
            x_train, y_train = self.transform_pair(rand_samples)
            x_train_all.append(x_train)
            #y_train_all.append(y_train)
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            #print "x_train: ", x_train
            #print "y_train: ", y_train
        x_train_all = np.vstack(x_train_all)
        #print "x_train_all: "
        #print x_train_all
        #print "y_train_all: "
        #print y_train_all
        self.svmr = SVMR().fit(x_train_all, y_train_all)
        return self.svmr.coef_.ravel()


    '''Function to train svm by using top k data points in terms of the query feature

    Args:
        asc (bool, optional): True if sorting query feature in ascending order. Meaning the lower the query feature value, the higher the input ranking
        k (int, optional): pick the top k data points to train svm

    Returns:
        weights of features: array
        
    '''
    def train_top_k(self, asc=False, k=-1):
        if k == -1:
            k = self.k
        #print 'k: ', k
        sorted_df = self.sort_dataframe(asc=asc, column=self.query_feature_i)
        top_k_ds = sorted_df[0:k]
        #print 'top k sorted_df:'
        #print top_k_ds
        x_train, y_train = self.transform_pair(top_k_ds)
        #print "x_train: ", x_train
        #print "y_train: ", y_train
        self.svmr = SVMR().fit(x_train, y_train)
        #print "weights: ", self.svmr.coef_.ravel()
        return self.svmr.coef_.ravel()

    '''Function to train svm by using top-1-k-quantile data points in terms of the query feature

    Args:
        asc (bool, optional): True if sorting query feature in ascending order: Meaning the lower the query feature value, the higher the input ranking
        k (int, optional): pick the top-1 from k quantiles data points to train svm

    Returns:
        weights of features: array
        
    '''
    def train_top_1_k(self, asc=False, k=-1):
        if k == -1:
            k = self.k
        sorted_df = self.sort_dataframe(asc=asc, column=self.query_feature_i)
        sample_inds = np.linspace(0, sorted_df.shape[0]-1, num=k, dtype=int)
        top_k_ds = sorted_df.iloc[sample_inds]
        #print "top_1_k: ", top_k_ds
        x_train, y_train = self.transform_pair(top_k_ds)
        #print "x_train: ", x_train
        #print "y_train: ", y_train
        self.svmr = SVMR().fit(x_train, y_train)
        #print "weights: ", self.svmr.coef_.ravel()
        return self.svmr.coef_.ravel()

    '''Function to get final ranking of all n data points 

    Args:
        asc (bool, optional): True if sorting query feature in ascending order: Meaning the lower the query feature value, the higher the input ranking

    Returns:
        descendingly sorted rank_df by ranking scores (i.e. SVMRank): dataframe, {'fips_code', <query feature>, 'SVMRank'}
        
    '''
    def get_full_rank(self, asc=False):
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
            return rank_df.sort_values('SVMRank', ascending=asc)
        else:
            raise ValueError("You must train the model before predicting!")

    '''
    Not a suitable test. Ground truth y values are purely query feature values. Just for reference.
    '''
    def score_test_query(self):
        if hasattr(self, 'svmr'):
            x = self.df.iloc[:, self.query_feature_i+1:]
            true_y = np.argsort(self.df.iloc[:, self.query_feature_i])
            score = self.svmr.score(x, true_y)
            #print "score: ", score
            return score
        else:
            raise ValueError("You must train the model before predicting!")

#***************HELPER FUNCTIONS*****************
        
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

def get_AP_k(rel, k):
    assert k>=1
    rel = rel[:k] != 0
    return np.mean(rel)

def get_AP(rel):
    rel = rel != 0
    out = [get_AP_k(rel, k+1) for k in range(rel.size) if rel[k]]
    if not out:
        return 0.
    return np.mean(out)
    
def experiments_k_i_ed():
    # [query, ranking1, ranking2 ..]
    num_trials = 10
    features = []
    features.append(['HEA010207D', 'BZN505206D', 'CLF040206D', 'EMN240205D', 'IPE010207D', \
 'HEA030206D', 'EMN370205D', 'AGE040206D', 'CRM260206D', 'BZN240206D',
 'EMN290206D', 'EMN012206D', 'BZN505208D', 'CRM210206D', 'BZA115208D'])
    rand_query = random.sample(range(0, len(features[0])), num_trials-1)
    for c in rand_query:
        focus = list(features[0])
        tmp = focus[c]
        focus[c] = focus[0]
        focus[0] = tmp
        features.append(focus)
    weights = []
# truth weights from iteratively training 50 random samples
    weights.append(np.array([-0.61481773, -0.01632002, 0.38787208, 0.06875974, 5.77053934 ,1.20733114, \
  9.55994288 ,-1.85361942 ,-0.11975094 ,-0.07411218 ,-0.37940481, -0.28949429, \
 -1.16763509, -0.01843858]))
    ks = np.array([5, 10, 15, 25, 50, 100])
    data = data_utils.load_data()
# prepare for truth rankings from the weights
    df_dict = {}
    true_ranks_dict = {}
    true_aps = []
    for k in xrange(len(features)):
        df_dict[k] = data_utils.get_features(features[k], data)
        #r = np.dot(df_dict[k].iloc[:, 2:], weights[k].T).ravel()
        #cols = [df_dict[k].columns[0]]
        #cols.append(df_dict[k].columns[1]) 
        #cols.append('SVMRank')
        #rank_df = pd.DataFrame(0.0, index=df_dict[k].index, columns=cols)
        #rank_df.iloc[:, :2] = df_dict[k].iloc[:, :2]
        #rank_df['SVMRank'] = r
        #true_ranks_dict[k] = rank_df.sort_values('SVMRank', ascending=False)
        #true_ranks = true_ranks_dict[k]
        #sorted_df = df_dict[k].sort_values(df_dict[k].columns[1], ascending=False)
        #cur_ap = get_AP(true_ranks['fips_code'].values==sorted_df['fips_code'].values)
        #true_aps.append(cur_ap)
    arr = np.zeros(shape=(2*ks.size, num_trials), dtype=float)
    for i in xrange(2):
        for j in xrange(ks.size):
            cur_k = ks[j]
            arr_i = i*ks.size+j
            for c in xrange(num_trials):
                svm = SVMRank(df_dict[c], cur_k)
                for t in xrange(num_trials):
                    ws = svm.train_top_k() if i==0 else svm.train_top_1_k()
                    ranks_df = svm.get_full_rank() 
                    sorted_df = df_dict[k].sort_values(df_dict[k].columns[1], ascending=False)
                    cur_ap = get_AP(sorted_df['fips_code'].values==ranks_df['fips_code'].values)
                    arr[arr_i][c] += cur_ap
                arr[arr_i][c] /= num_trials
    print 'AP array(i*k*c): '
    for i in arr:
        print ",".join(str(v) for v in i)
    print 'MAPs for top-k and top-1-k: ', np.mean(arr, axis=1)
    print 'MAPs bw ground truth and true', true_aps


def experiments_k_i():
    # [query, ranking1, ranking2 ..]
    features = []
    features.append(['EDU015208D', 'EMN012207D', 'EAN450207D', 'CLF040209D', 'CLF010209D'])
#    features.append(['CRM210208D', 'EDU695209D', 'FED110208D', 'HEA040207D', 'HEA270208D', 'EMN290207D'])
#    features.append(['BNK150209D', 'AGE040209D', 'BZA210209D', 'CLF040209D'])
    features_asc = [False, True, False]
    weights = []
# truth weights from iteratively training 50 random samples
#    weights.append(np.array([-0.10839324, -2.68457891, 0.11246163, 12.60964627]))
#    weights.append(np.array([-1.41442939, 2.74791654, 1.81398451, -0.35598531, 0.51140667]))
#    weights.append(np.array([3.43737945, 2.96887991, 0.01150457]))
# truth weights from 2000 random samples
    weights.append(np.array([0.42236209, -1.34034753, 0.10839856, 9.35157845]))
    weights.append(np.array([-1.252217, 2.55608963, 1.61006832, -0.35262227, 0.46754842]))
    weights.append(np.array([3.70088954, 3.26022662, 0.0143878]))
    num_trials = 2
    ks = np.array([5, 10, 15, 25, 50, 100, 200, 300])
    data = data_utils.load_data()
# prepare for truth rankings from the weights
    df_dict = {}
    true_ranks_dict = {}
    for k in xrange(len(features)):
        df_dict[k] = data_utils.get_features(features[k], data)
        #print 'data frame: '
        #print df_dict[k]
        r = np.dot(df_dict[k].iloc[:, 2:], weights[k].T).ravel()
        cols = [df_dict[k].columns[0]]
        cols.append(df_dict[k].columns[1]) 
        cols.append('SVMRank')
        rank_df = pd.DataFrame(0.0, index=df_dict[k].index, columns=cols)
        rank_df.iloc[:, :2] = df_dict[k].iloc[:, :2]
        rank_df['SVMRank'] = r
        true_ranks_dict[k] = rank_df.sort_values('SVMRank', ascending=False)
        #print 'true rank frame: '
        #print true_ranks_dict[k]
    arr = np.ndarray(shape=(2, ks.size), dtype=float)
    dist_arr = np.ndarray(shape=(2, ks.size), dtype=float)
    for i in xrange(2):
        for j in xrange(ks.size):
            ap_sum = 0.
            cur_k = ks[j]
            ap_list = []
            w_sum = 0.
            for k in xrange(len(features)):
                svm = SVMRank(df_dict[k], cur_k)
                true_ranks = true_ranks_dict[k]
                for c in xrange(num_trials):
                    ws = svm.train_top_k() if i==0 else svm.train_top_1_k()
# find ED between exp. weights and true weights
                    w_sum += np.linalg.norm(ws-weights[k])
                    ranks_df = svm.get_full_rank() 
                    #print true_ranks['fips_code'].values
                    #print ranks_df['fips_code'].values
                    cur_ap = get_AP(true_ranks['fips_code'].values==ranks_df['fips_code'].values)
                    ap_list.append(cur_ap)
                    ap_sum += cur_ap 
            arr[i, j] = ap_sum / len(features)/num_trials
            dist_arr[i, j] = w_sum / len(features)/num_trials
            print str(i)+"way "+str(cur_k)+"k "+"_aps: ", ap_list
    print 'AP array(i*k): '
    print arr
    print 'MAPs for top-k and top-1-k: ', np.mean(arr, axis=1)
    print 'MAPs for ks', np.mean(arr, axis=0)
    print 'Euclidean Distances'
    print 'ED array(i*k): '
    print dist_arr
    print 'EDs for top-k and top-1-k: ', np.mean(dist_arr, axis=1)
    print 'EDs for ks', np.mean(dist_arr, axis=0)

    
def train_ground_truth_wts():
    data = data_utils.load_data()
    #query_feature = ['EDU015208D']
    #subset_features = query_feature + ['EMN012207D','EAN450207D','CLF040209D','CLF010209D']
    #query_feature = ['CRM210208D']
    #subset_features = query_feature + ['EDU695209D','FED110208D','HEA040207D','HEA270208D', 'EMN290207D']
    #query_feature = ['BNK150209D']
    #subset_features = query_feature + ['AGE040209D','BZA210209D','CLF040209D']
    subset_features = ['HEA010207D', 'BZN505206D', 'CLF040206D', 'EMN240205D', 'IPE010207D', \
 'HEA030206D', 'EMN370205D', 'AGE040206D', 'CRM260206D', 'BZN240206D',
 'EMN290206D', 'EMN012206D', 'BZN505208D', 'CRM210206D', 'BZA115208D']
    focus_frame = data_utils.get_features(subset_features,data)
    svm = SVMRank(focus_frame, k=50)
    print svm.train_random_k_iter(iterations=1000,k=50)

def get_random_features(n):
    data = data_utils.load_data()
    total = len(data.columns)
    rand_inds = random.sample(range(0, total), n)
    rand_samples = data.columns.values[rand_inds]
    print rand_samples
    

if __name__=='__main__':
#    get_random_features(15)
#    train_ground_truth_wts()
    experiments_k_i_ed()
#    experiments_k_i()
#    data = data_utils.load_data()
#    #query_feature = ['EDU015208D']
#    #subset_features = query_feature + ['EMN012207D','EAN450207D','CLF040209D','CLF010209D']
#    query_feature = ['CRM210208D']
#    subset_features = query_feature + ['EDU695209D','FED110208D','HEA040207D','HEA270208D', 'EMN290207D']
#    #query_feature = ['BNK150209D']
#    #subset_features = query_feature + ['AGE040209D','BZA210209D','CLF040209D']
#    focus_frame = data_utils.get_features(subset_features,data)
#    #sorted_frame = focus_frame.sort_values(query_feature[0], ascending=False)
#    #print sorted_frame
#    svm = SVMRank(focus_frame, 100)
#    #s = time.time()
#    weights_k = svm.train_random_k()
#    #print 'finished training top k in ', str(time.time()-s)
#    print 'weights_k: ', weights_k 
#    frame_ranks_k = svm.get_full_rank()
#    #ap = get_AP(frame_ranks_k[query_feature[0]]==frame_ranks_k['SVMRank'])
#    #s = time.time()
#    weights_1k = svm.train_top_1_k()
#    #print 'finished training top 1k in ', str(time.time()-s)
#    print 'weights_1k: ', weights_1k 
#    frame_ranks_1k = svm.get_full_rank()
#    #score = svm.score_test_query()
#    print frame_ranks_k
#    print frame_ranks_1k
