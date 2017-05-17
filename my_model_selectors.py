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

    def base_model(self, num_states, X, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
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
        return self.base_model(best_num_components, self.X, self.lengths)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score = float('inf')
        best_model = None

        # Number of elements
        p = len(self.lengths)

        for n in range(self.min_n_components, self.max_n_components):
            try:
                logN = math.log(n)
                model = self.base_model(n, self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
            except ValueError:
                # Bug with hmmlearn for large N
                continue
            except AttributeError:
                # Model couldn't be trained
                continue

            score = -2 * logL + p * logN

            if score < best_score:
                best_score = score
                best_model = model

        return best_model if best_model else self.base_model(self.n_constant, self.X, self.lengths)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)

        best_score = float('-inf')
        best_model = None

        # score = (M-1) * DIC = [(M-1)* logL of against current word] - Sum(LogL against other words)
        # = M * (logL against current word) - <logL against all words>
        # Let's avoid computing logL against current word twice though, even if it's more convenient

        # Word count
        word_count = len(self.lengths)

        # We can't work with less than 2 words
        if word_count < 2:
            return self.base_model(self.n_constant, self.X, self.lengths)

        for n in range(self.min_n_components, self.max_n_components):
            try:
                # Generate the model
                model = self.base_model(n, self.X, self.lengths)

                if not model:
                    continue

                # Score the model
                score = 0
                for word, (x, lengths) in self.hwords.items():
                    # print("Scoring against {} and {}".format(x, lengths))
                    test_score = model.score(x, lengths)
                    multiplier = -1 if word != self.this_word else (word_count - 1)
                    score += test_score * multiplier

            except ValueError:
                # Bug with hmmlearn for large N
                continue

            if score > best_score:
                best_score = score
                best_model = model

        return best_model if best_model else self.base_model(self.n_constant, self.X, self.lengths)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Special handling when there's only 1 sample
        # print("There are {} samples".format(len(self.lengths)))
        if len(self.lengths) == 1:
            return self.base_model(1, self.X, self.lengths)

        # print("Splits = {}".format(len(self.lengths)))
        split_method = KFold(min(len(self.lengths), 3))

        # This should never stay the same, we hope
        best_n = self.random_state
        best_avg_log_l = float('-inf')

        for n in range(self.min_n_components, self.max_n_components):
            # print('N = {}'.format(n))
            # Create all the models with n components for cross validation
            total_logLs = 0
            count = 0
            for train_indexes, test_indexes in split_method.split(self.sequences):
                try:
                    # print("Train indexes: {} Test indexes: {}".format(train_indexes, test_indexes))
                    # Let's do some training!
                    (train_X, train_lengths) = combine_sequences(train_indexes, self.sequences)
                    model = self.base_model(n, train_X, train_lengths)

                    # If we cannot train the model
                    if not model:
                        break  # Stop trying and navigate to the outer loop already, count should be 0

                    # Now test the fire on this sub model
                    (test_X, test_lengths) = combine_sequences(test_indexes, self.sequences)
                    logL = model.score(test_X, test_lengths)
                    total_logLs += logL
                    count += 1
                except ValueError:
                    # Bug with hmmlearn for large N and not enough data frames
                    pass

            if count > 0:
                avgLogL = total_logLs / count
                if avgLogL > best_avg_log_l:
                    best_n = n
                    best_avg_log_l = avgLogL
                    # print('Best average logL {} with N={}'.format(best_avg_log_l, best_n))

        # Now extract the best N and generate a model with all the training data we have
        return self.base_model(best_n, self.X, self.lengths) if best_n > 0 else None
