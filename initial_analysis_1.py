#!/usr/bin/python

#initial analysis on commit comments and pull request comments
import sys

import pandas as pd
import numpy as np
import nltk
from collections import Counter
import re
import itertools
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ComputeClusters():
    def __init__(self, X_train = DictVectorizer(), X_test = DictVectorizer()):
        self.X_train = X_train
        self.X_test = X_test
    def kmeans_fit(self):
        self.kmeans = KMeans(n_clusters=117).fit(X_train)
    def kmeans_predict(self):
        return self.kmeans.predict(X_test)
    def kmeans_cluster(self):
        return self.kmeans.cluster_centers_

class ExtractFeatures():
    def __init__(self):
        self.df = pd.DataFrame()
        self.features = Counter()
    def add_df(self, df = pd.DataFrame()):
        self.df = df
    def ngrams(self, tokens, n):
        output = []
        for i in range(n-1, len(tokens)):
            ngram = ' '.join(tokens[i-n+1:i+1])
            output.append(ngram)
        return output
    def skipgrams(self, tokens, n):
        output = []
        for i in range(n-1, len(tokens)):
            str1 = ''
            for j in range(n-2):
                str1 = str1+' * '
            skipgram = str(tokens[i-n+1])+str1+str(tokens[i])
            output.append(skipgram)
        return output
    def pairs(self, words):
        ret = itertools.combinations(words,2)
        return ret
    def parts_of_speech(self, words):
            tagged = []
            try:
                tagged = nltk.pos_tag(words)
                return(tagged)
        
            except Exception as e:
                 print(e)
    #preconditions: use on android code and android stack traces only
    def extract_code_trace_features(self, title, trace_text):
        features = Counter()
    
        trace_lines_re = re.compile('(android\.[\w\.$]*)')
        trace_lines = trace_lines_re.findall(trace_text)
    
        features += Counter(trace_lines)
    
        in_brackets_re = re.compile('android\.[\w\.$]*\(([\S]*):(?![ ])')
        in_brackets = in_brackets_re.findall(trace_text)
    
        features += Counter(in_brackets)
    
        return features
    def extract_text_features(self, discussion_text, title = ''):
        discussion_text = discussion_text.lower()
        title = title.lower()
    
        discussion_text = re.sub(r'(.)\1+', r'\1\1', discussion_text) #remove more than 2 consecutive repeated characters
        title = re.sub(r'(.)\1+', r'\1\1', title)
    
        features_in_text = []
    
        text_alphanum = re.sub('[^a-z0-9]', ' ', discussion_text)
        title_alphanum = re.sub('[^a-z0-9]', ' ', title)
    
        for n in range(3,7):
            features_in_text += self.ngrams(text_alphanum.split(), n)
        
        for n in range(3,7):
            features_in_text += self.ngrams(title_alphanum.split(), n)
        
        for n in range(3,7):
            features_in_text += self.skipgrams(text_alphanum.split(), n)
        
        for n in range(3,7):
            features_in_text += self.skipgrams(title_alphanum.split(), n)
        
            features_in_text += str(self.pairs(text_alphanum.split()))
            features_in_text += str(self.pairs(title_alphanum.split()))
    
            features_in_text += str(self.parts_of_speech(text_alphanum.split()))
            features_in_text += str(self.parts_of_speech(title_alphanum.split()))
    
            return Counter(features_in_text)

    def strip_code(self, text):
    
        brackets_re = re.compile('([^\n]*{[^}]*})')
        text = brackets_re.sub('',text)
    
        linestrip_re = re.compile('([^\n]*[;()])')
        text = linestrip_re.sub('',text)
    
        urlstrip_re = re.compile('http[s]*:\/\/[\S]*')
        text = urlstrip_re.sub('',text)
    
        return text
    def strip_all_code(self):
        for row in range(len(self.df)):
            self.df['body'].iloc[row] = self.strip_code(self.df['body'].iloc[row])
    def dict_vectorize_text_features(self):
        vect = DictVectorizer()
        features = Counter()
        for row in range(len(self.df)):
            text = self.df['body'].iloc[row]
            features += self.extract_text_features(text)
        self.features = features
        X = vect.fit_transform(features)
        feature_names = vect.feature_names_
        X = np.reshape(X,newshape=(len(features),-1))
        X_train, X_test = train_test_split(X, test_size=0.4, train_size=0.6, random_state=0)
        return [X_train, X_test, features, feature_names]
    def get_df(self):
        return self.df
    def get_indices(self):
        indices = {}
        for row in range(len(self.df)):
            body = self.df['body'].iloc[row]
            features = self.extract_text_features(body)
            for feature in features:
                if feature in indices:
                    indices[feature].append(row)
                else:
                    indices[feature] = [row]
        return indices

        
class ProcessDataframe():
    def __init__(self):
        self.df = pd.DataFrame()
    def add_df(self, df = pd.DataFrame()):
        self.df = df
    def find_missing(self):
        nan = 0
        float_ = 0
        no_text = []
        excepts = 0

        for row in range(len(self.df)):
            try:
                if self.df['body'].iloc[row] == ('NaN' or 'nan' or np.nan):
                    print('first if')
                    nan += 1
                    no_text.append(row)
                elif type(self.df['body'].iloc[row]) == float:
                    print('second if')
                    float_ += 1
                    no_text.append(row)
            except:
                excepts += 1
                no_text.append(row)
            
        return [nan, float_, no_text, excepts]
    def remove_all_missing(self):
        l = self.find_missing()
        print('missing')
        print(len(l[2]))
        print(l[3])
        no_text = l[2]
        while (len(no_text) > 0):
            for index in no_text:
                if index < len(self.df):
                    self.df = self.df.drop(self.df.index[index])
            l = self.find_missing()
            print('missing')
            print(len(l[2]))
            print(l[3])
            no_text = l[2]
    def get_df(self):
        return self.df
    def convert_str(self):
        for row in range(len(self.df)):
            self.df['body'].iloc[row] = str(self.df['body'].iloc[row])


if __name__=="__main__":
    print('running')
    df1 = pd.read_csv('/Volumes/Elements/mysql-2019-06-01/commit_comments.csv', header=None, sep=',', usecols = [3], nrows =5000)
    #df2 = pd.read_csv('/Volumes/Elements/mysql-2019-06-01/pull_request_comments.csv', header=None, sep=',', usecols = [4])
    print('read dataframes')
    df1.columns = ['body']
    #df2.columns = ['body']

    #df = df1.append(df2, ignore_index=True)

    df_sample = df1
    print(len(df_sample))

    process = ProcessDataframe()
    print('initialized process')
    process.add_df(df_sample)
    print('added dataframe')
    print('converting')
    process.convert_str()
    print('converted')
    print('removing missing')
    process.remove_all_missing()
    print('removed missing')
    print('processed dataframes')
    df2 = process.get_df()
    print(len(df2))
    features = ExtractFeatures()
    features.add_df(df2)
    print('added df')
    features.strip_all_code()
    print('stripped code')
    X_train, X_test, feature_counter, feature_names = features.dict_vectorize_text_features()
    print('vectorized text features and split data')
    clusters = ComputeClusters(X_train, X_test)
    print('created cluster object')
    clusters.kmeans_fit()
    print('fitted')
    clusters.kmeans_predict()
    print('predicted')
    print('calculated clusters')
    centers = clusters.kmeans_cluster()
    print(centers)
    print('dataframe indices')
    indices = features.get_indices()
    print('got indices')

    name_dict ={}
    for i in range(len(feature_names)):
        name_dict[feature_names[i]] = i

    reference = {}
    for i in feature_counter:
        if i in indices:
            row = indices[i]
            for r in row:
                centers_ = {}
                body = df1['body'].iloc[r]
                for center in centers:
                    diff = abs(name_dict[i]-int(center))
                    centers_[diff] = int(center)
                center = min(centers_)
                if center in reference:
                    reference[center].add(body)
                else:
                    reference[center] = set(body)

    file = open("clusters117.txt","w+")
    for entry in reference:
        file.write('\n\nCluster:' + str(entry))
        file.write(str(reference[entry]))
        file.write('\n')



    
