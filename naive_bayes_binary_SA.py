import numpy as np
import math
import nltk
import pandas as pd
import random
from nltk.corpus import movie_reviews

class NaiveBayesClassifierBinary:

    def __init__(self):
        self.log_prior = {}
        self.pos_words = {}
        self.neg_words = {}
        self.log_likelihood_pos = {}
        self.log_likelihood_neg = {}

        

    def calculate_frequency(self, training_data):
        # 각 Class 빈도 계산
        num_pos = 0
        num_neg = 0
        exceptions = [',', '.', 't']
        for docu in training_data:
            docu_word_pos = {}
            docu_word_neg = {}
            if docu[1] == 'pos':
                num_pos += 1
                for word in docu[0]:
#                     if len(word) == 1 and word not in exceptions:
#                         continue
                    if word in self.pos_words:
                        if word not in docu_word_pos:
                            self.pos_words[word] += 1
                            docu_word_pos[word] = 1
                    else:
                        self.pos_words[word] = 1
            else:
                num_neg += 1
                for word in docu[0]:
                    if word in self.neg_words:
                        if word not in docu_word_neg:
                            self.neg_words[word] += 1
                            docu_word_neg[word] = 1
                    else:
                        self.neg_words[word] = 1
        
        return (num_pos, num_neg)
                    
    def log_likelihood(self):
        count_all_pos = 0
        count_all_neg = 0
        for word in self.pos_words:
            count_all_pos = count_all_pos + 1
        
        for word in self.neg_words:
            count_all_neg = count_all_neg + 1
            
        for word in self.pos_words:
            self.log_likelihood_pos[word] = math.log2((self.pos_words[word])  
                                                / count_all_pos)
        
        for word in self.neg_words:
            self.log_likelihood_neg[word] = math.log2((self.neg_words[word]) 
                                                / count_all_neg)
        
        self.smoothing_pos = math.log2(1 / count_all_pos)
        self.smoothing_neg = math.log2(1 / count_all_neg)

        
        
    def train(self, training_data):
        num_doc = len(training_data)
        print('num doc : %d' %(num_doc))
        num_pos, num_neg = self.calculate_frequency(training_data)
        
        self.log_prior['pos'] = math.log2(num_pos / num_doc)
        self.log_prior['neg'] = math.log2(num_neg / num_doc)
        
        self.log_likelihood()
        
    def predict(self, test_data):
        sum_pos = self.log_prior['pos']
        sum_neg = self.log_prior['neg']
        word_in_docu_pos = {}
        word_in_docu_neg = {}

        for word in test_data[0]:
            if word in self.log_likelihood_pos:
                if word in word_in_docu_pos:
                    continue
                else:
                    sum_pos += self.log_likelihood_pos[word]
                    word_in_docu_pos[word] = 1
            else:
                if not word in word_in_docu_pos:
                    sum_pos += self.smoothing_pos
                    word_in_docu_pos[word] = 1
            if word in self.log_likelihood_neg:
                if word in word_in_docu_neg:
                    continue
                else:
                    sum_neg += self.log_likelihood_neg[word]
                    word_in_docu_neg[word] = 1
            else:
                if not word in word_in_docu_neg:
                    sum_neg += self.smoothing_neg
                    word_in_docu_neg[word] = 1

        if sum_pos > sum_neg:
            return 'pos'
        else:
            return 'neg'

    def accuracy(self, test_data):
        data_size = len(test_data)
        correct_num = 0
        for docu in test_data:
            if self.predict(docu) == docu[1]:
                correct_num += 1
                
        return float(correct_num / data_size)
                
                
    
                
    
def fold_10(set_of_documents):
    """
    Simple 10 folds CV
    """
    size = len(set_of_documents)
    index_interval = int(size / 10)
    
    ten_folds = []
    for i in range(10):
        start_index = int(i * index_interval)
        if i == 9:
            ten_folds.append(set_of_documents[start_index:])
        else:
            ten_folds.append(set_of_documents[start_index:start_index + index_interval])
    
    res = []
    for i in range(10):
        _test_set = ten_folds[i]
        _training_set = []
        for j in range(10):
            if j == i:
                continue
            for elem in ten_folds[j]:
                _training_set.append(elem)
        ith_fold = [_training_set, _test_set]
        res.append(ith_fold)
    
    return res


def main():
    documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    # Random Shuffle with the random seed
    random.seed(19941225)
    random.shuffle(documents)

    accuracy_list = []
    iter_ = 0

    for training_test in fold_10(documents):
        
        training_data = training_test[0]
        test_data = training_test[1]
        
        naive_bayes = NaiveBayesClassifierBinary()
        naive_bayes.train(training_data)
        
        accuracy = naive_bayes.accuracy(test_data)

        accuracy_list.append(accuracy)
        iter_ += 1
        
    print(accuracy_list)
    print('average : ', (sum(accuracy_list) / len(accuracy_list)))

if __name__ == '__main__':
    main()
