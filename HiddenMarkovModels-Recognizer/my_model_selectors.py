import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = None
        best_model = None

        if self.verbose:
            print('BIC. Selecting best model for word: {}'.format(self.this_word))
        
        i = self.min_n_components
        
        try:
            while i <= self.max_n_components:
                                                
                model = self.base_model(i)
                
                if model is not None:
                    bic = self.get_bic(model)
                    
                    if self.verbose:
                        print('State: {}, BIC: {}'.format(i, bic))
                    
                    if (best_model is None) or (bic < best_score):
                        best_model = model
                        best_score = bic
                
                # move to next state
                i += 1
        except Exception as e:
            if self.verbose:
                print('Terminated BIC selector at {} for word \'{}\'. Error: {}'.format(i, self.this_word, e))
                
        if self.verbose:
            print('Word: {}, Best State: {}'.format(self.this_word, (i-1)))
        
        return best_model

     
    def get_bic(self, model):
        '''
        Get the BIC score of the HMM model
        '''
        m = model.n_components
        k = 2 * len(model.means_[0])
        p = m**2 + k*m -1
        logL = model.score(self.X, self.lengths)
          
        return (-2.0 * logL + p * np.log(m))
        

class SelectorDIC(ModelSelector):
    ''' 
    Select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = None
        best_model = None
        
        if self.verbose:
            print('DIC. Selecting best model for word: {}'.format(self.this_word))
        
        i = self.min_n_components
        
        try:
            while i <= self.max_n_components:
                                                
                model = self.base_model(i)
                
                if model is not None:
                    dic = self.get_dic(model)
                    
                    if self.verbose:
                        print('State: {}, DIC: {}'.format(i, dic))
                    
                    if (best_model is None) or (dic > best_score):
                        best_model = model
                        best_score = dic
                    
                # move to next state
                i += 1
        except Exception as e: 
            
            if self.verbose:
                print('Terminated DIC selector at {} for word \'{}\'. Error: {}'.format(i, self.this_word, e))
        
        if self.verbose:
            print('Word: {}, Best State: {}'.format(self.this_word, (i-1)))
        
        return best_model
            
        
    def get_dic(self, model):
        '''
        Get the DIC score of the HMM model
        '''
        states = model.n_components
                        
        # get log loss for current model
        logL = model.score(self.X, self.lengths)
        
        other_words = list(self.words.keys())
        word_count = len(other_words)
        other_words.remove(self.this_word)
                        
        other_logL = 0;
                
        # calculate log loss of other words
        for word in other_words: 
            try: 
                if len(self.words[word]) > 2:
                    other_model = ModelSelector(self.words, self.hwords, word, self.n_constant,
                                        self.min_n_components, self.max_n_components, 
                                        self.random_state, self.verbose)
                    
                    hmm = other_model.base_model(states)
                    other_logL += hmm.score(other_model.X, other_model.lengths)
            except:
                word_count -= 1
        
        # get dic
        dic = logL - other_logL / (word_count -1)
        
        return dic
    
    
class SelectorCV(ModelSelector):
    ''' 
    select best model based on average log Likelihood of cross-validation folds
    
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_score = None
        best_state = self.min_n_components
        
        i = self.min_n_components
        
        if self.verbose:
            print('CV. Selecting best model for word: {}'.format(self.this_word))
        
        try:
            while i <= self.max_n_components:
                                                
                cv_score = self.get_cv_score(i)
                
                if self.verbose:
                    print('State: {}, CV Score: {}'.format(i, cv_score))
                
                if best_score is None:
                    best_state = i
                    best_score = cv_score
                elif cv_score > best_score:
                    best_state = i
                    best_score = cv_score
                    
                # move to next state
                i = i + 1
        except Exception as e:           
            if self.verbose:
                print('Terminated CV selector at {} for word \'{}\'. Error: {}'.format(i, e, self.this_word))
        
        if self.verbose:
            print('Word: {}, Best State: {}'.format(self.this_word, best_state))
        
        best_model = self.base_model(best_state)
        
        return best_model   
                
    def get_cv_score(self, states):
        """
        Get the cross validation score for the model
        """
        
        folds = 3
        if len(self.sequences) < folds:
            folds = len(self.sequences)
            
        split_method = KFold(n_splits = folds)
                        
        logL = 0.0
        splits = 0
        
        for train_index, test_index in split_method.split(self.sequences):
            
            X, lengths = combine_sequences(train_index, self.sequences)
            
            model = ModelSelector(self.words, self.hwords, self.this_word, self.n_constant,
                                  self.min_n_components, self.max_n_components, 
                                  self.random_state, self.verbose)            
            model.X       = X
            model.lengths = lengths
                                    
            hmm   = model.base_model(states)
            logL += hmm.score(model.X, model.lengths)
            splits += 1
                 
        return (logL / splits)
       