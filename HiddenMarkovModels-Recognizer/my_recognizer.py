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
    guesses = []
    
    for word_id in range(len(test_set.wordlist)):
        
        probability = None
        prediction  = None
                
        X, lengths = test_set.get_item_Xlengths(word_id)
        
        word_probs = {}
        
        for word in list(models.keys()):
            model = models[word]
            
            if model is not None:
                
                try:
                    logL = model.score(X, lengths)
                    
                    word_probs[word] = logL
                    
                    if (probability is None) or (logL > probability):
                        probability = logL
                        prediction = word  
                except Exception as e:
                    pass
        
        probabilities.append(word_probs)
        guesses.append(prediction)
       
    return probabilities, guesses
