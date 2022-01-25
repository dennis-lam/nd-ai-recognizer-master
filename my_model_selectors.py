import math
import statistics
import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


def find_min_value_index(list):
    """
    Find the minimum value index from list
    """
    min_index = None
    min_value = None
    for index, value in enumerate(list):
        if min_value is None or min_value > value:
            min_index = index
            min_value = value
    return min_index


def find_max_value_index(list):
    """
    Find the maximum value index from list
    """
    max_index = None
    max_value = None
    for index, value in enumerate(list):
        if max_value is None or value > max_value:
            max_index = index
            max_value = value
    return max_index

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
    # Reference: Udacity Forum 233235
    def get_n_parameters(self, n_components):
        """
        Find number of parameters
        """
        n_features = self.X.shape[1]
        tran_prob = n_components * (n_components - 1)
        starting_prob = n_components - 1
        n_means = n_components * n_features
        n_variances = n_components * n_features
        return tran_prob + starting_prob + n_means + n_variances

    def get_score(self, n_components):
        """
        Calculate BIC score
        """
        try:
            model = self.base_model(n_components)
            logL = model.score(self.X, self.lengths)
            p = self.get_n_parameters(n_components)
            logN = np.log(sum(self.lengths))
            return -2 * logL + p * logN
        except Exception as e:
            if self.verbose:
                print(e)
            # Return self-define max value.
            return 9999999

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # component number list.
        n_components_list = range(self.min_n_components, self.max_n_components + 1)
        # Obtain score collection.
        score_list = [self.get_score(n_components) for n_components in n_components_list]
        # Find the min score index
        min_score_index = find_min_value_index(score_list)
        # Returns HMM
        return self.base_model(n_components_list[min_score_index])

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def get_logL(self, model, word):
        """
        Obtain score from model and word
        """
        X, lengths = self.hwords[word]
        return model.score(X, lengths)


    def get_score(self, n_components):
        """
        Calculate DIC score
        """
        try:
            model = self.base_model(n_components)
            this_word_logL = self.get_logL(model, self.this_word)
            others_words = [word for word in self.words if word != self.this_word]
            others_words_average_logL = statistics.mean([self.get_logL(model, word) for word in others_words])
            return this_word_logL - others_words_average_logL
        except Exception as e:
            if self.verbose:
                print(e)
            # Return self-define min value.
            return -9999999

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # component number list.
        n_components_list = range(self.min_n_components, self.max_n_components + 1)
        # Obtain score collection.
        score_list = [self.get_score(n_components) for n_components in n_components_list]
        # Find the max score index
        max_score_index = find_max_value_index(score_list)
        # Returns HMM
        return self.base_model(n_components_list[max_score_index])


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def get_score(self, n_components, train_indices, test_indices):
        """
        Obtain score from particular Fold
        """
        try:
            train_X, train_lengths = combine_sequences(train_indices, self.sequences)
            test_X, test_lengths = combine_sequences(test_indices, self.sequences)
            model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
            return model.score(test_X, test_lengths)
        except Exception as e:
            if self.verbose:
                print(e)
            # Return self-define min value.
            return -9999999

    def get_average_score(self, n_components):
        """
        Obtain average score from K-Fold
        """
        # Initialize K-Fold
        max_splits = len(self.lengths)
        n_splits = max_splits if max_splits < 3 else 3
        if n_splits > 1:
            split_method = KFold(n_splits=n_splits, random_state=self.random_state)
            # Obtain score list
            score_list = [self.get_score(n_components, train_indices, test_indices)
                          for train_indices, test_indices in split_method.split(self.sequences)]
            # Find the mean score.
            return statistics.mean(score_list)
        else:
            # Just return score if contains no splits
            return self.get_score(n_components, [0], [0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # component number list.
        n_components_list = range(self.min_n_components, self.max_n_components + 1)
        # Obtain score collection.
        score_list = [self.get_average_score(n_components) for n_components in n_components_list]
        # Find the max score index
        max_score_index = find_max_value_index(score_list)
        # Returns HMM
        return self.base_model(n_components_list[max_score_index])

