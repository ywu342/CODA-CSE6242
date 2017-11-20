import pandas as pd
class SVMRank(object):
    def __init__(self, df, k, query_feature=df.columns[1]):
        self.df = df
        self.k = k
        self.query_feature = query_feature

    '''Function to sort the whole dataframe given a column    

    Args:
        column (str): sorting based on this column

    Returns:
        A sorted dataframe.
        
    '''
    def sort_dataframe(self, column, asc=True):
        return self.df.sort_values(column, ascending=asc)

    def train_top_k(self, k=self.k):
        sorted_df = sort_column()


