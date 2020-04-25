from data_preprocessing.data_preprocessing import DataPreprocessing
import re
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from bs4 import BeautifulSoup
from nltk import tokenize

class GRUATTDataPreprocessing(DataPreprocessing):

    def __init__(self):
        super(GRUATTDataPreprocessing, self).__init__()

    def process(self, df, test_options):
        file_name = 'data/gruatt_test_processed.csv' if test_options else 'data/gruatt_train_processed.csv'
        if os.path.isfile(file_name):
            return pd.read_pickle(file_name)
        for col in ['project_title', 'project_resource_summary', 'project_essay_1',
                    'project_essay_2', 'project_essay_3', 'project_essay_4']:
            df[col].fillna('', inplace=True)
        df['text'] = df.apply(
            lambda row: ' '.join(
                [str(row['project_title']), str(row['project_resource_summary']), str(row['project_essay_1']),
                 str(row['project_essay_2']), str(row['project_essay_3']), str(row['project_essay_4'])]), axis=1)
        df = df.drop(['teacher_id', 'teacher_prefix', 'school_state', 'project_submitted_datetime',
                      'project_grade_category', 'project_subject_categories', 'project_subject_subcategories',
                      'project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4',
                      'project_resource_summary', 'teacher_number_of_previously_posted_projects'], axis=1)
        print('before processing')
        rand_index = np.random.randint(len(df))
        print(df.iloc[rand_index].values)
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(self.process_df, np.array_split(df, mp.cpu_count()))
            pool.close()
            pool.join()
        df = pd.concat(result)
        print('\n')
        print('after processing')
        print(df.iloc[rand_index].values)
        df.to_pickle(file_name)
        return df

    def process_df(self, df):
        df['text'] = df['text'].apply(self.clean_sentences)
        return df

    def clean_sentences(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = super(GRUATTDataPreprocessing, self).clean_sentences(string)
        string = tokenize.sent_tokenize(string)
        return string