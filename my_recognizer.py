import warnings
from asl_data import SinglesData

def find_max_value_word(dict):
    """
    Find the maximum word index from dictionary
    """
    max_word = None
    max_value = None
    for word, value in dict.items():
        if max_value is None or value > max_value:
            max_word = word
            max_value = value
    return max_word

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

    def get_model_score(model, X, lengths):
        """
        Obtain model score without exception
        """
        try:
            return model.score(X, lengths)
        except Exception as e:
            # Return self-define min value.
            return -9999999

    def get_item_result(Xlengths):
        """
        Obtain result from particular item
        """
        X, lengths = Xlengths
        item_probabilities = {word : get_model_score(model, X, lengths) for word, model in models.items()}
        return item_probabilities, find_max_value_word(item_probabilities)

    # Handle each testing sample
    for index in test_set.get_all_Xlengths():
        item_probabilities, item_guess =  get_item_result(test_set.get_item_Xlengths(index))
        probabilities.append(item_probabilities)
        guesses.append(item_guess)

    return probabilities, guesses