import warnings
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
    guesses = ["" for x in range(test_set.num_items)]
    # guesses = np.full(test_set.num_items, ' ') # Empty string doesn't work here. Probably numpy bug.
    word_probabilities = [float('-inf') for x in range(test_set.num_items)]

    # Let's build the probabilities first
    for word, model in models.items():
        model_probabilities = {}  # Probabilities for this model/word
        for item in range(test_set.num_items):
            # Score this test sequence
            x, lengths = test_set.get_item_Xlengths(item)  # There's only one X,length in each test item
            try:
                logL = model.score(x, lengths)
                model_probabilities[word] = logL

                # Update the guesses
                if logL > word_probabilities[item]:
                    word_probabilities[item] = logL
                    guesses[item] = word
            except ValueError:
                # HMM bug
                pass

        # Add tho the list of model probabilities
        probabilities.append(model_probabilities)

    # print(probabilities, guesses)
    return (probabilities, guesses)
