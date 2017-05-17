import warnings
import numpy as np
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for item in range(test_set.num_items):
        current_probabilities = {}
        best_probability = float('-inf')
        best_guess = ""
        for word, model in models.items():
            print("Model for {} is {}".format(word, model))
            x, lengths = test_set.get_item_Xlengths(item)
            current_probabilities[word] = float('-inf')
            try:
                logL = model.score(x, lengths)
                current_probabilities[word] = logL
                if logL > best_probability:
                    best_probability = logL
                    best_guess = word
            except ValueError:
                # HMM Bug
                pass

        probabilities.append(current_probabilities)
        guesses.append(best_guess)

    return (probabilities, guesses)
