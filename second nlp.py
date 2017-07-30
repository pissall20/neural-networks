import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

# Creates a dictionary of words with the count as well
def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                # Tokenize every input line and add to the lexicon list
                all_words = word_tokenize(l.lower().decode('utf8'))
                lexicon += list(all_words)

    # Find the lemma for all words in the lexicon
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Count the words
    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        # Words used more than 50 times and less than 1000 times to avoid commonly used words and rare words
        if 50 < w_counts[w] < 1000:
            l2.append(w)
    print len(l2)
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            # Tokenize
            current_words = word_tokenize(l.lower().decode('utf8'))
            # Lemmatize
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # Create a blank feature set and increment the value for every word
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    # Get index of current word
                    index_value = lexicon.index(word.lower().decode('utf8'))
                    # Increment the feature value at the index
                    features[index_value] += 1
            features = list(features)
            # Append the features and the labels to the list
            featureset.append([features, classification])

    return featureset

# Creates features and labels out of the samples
def create_feature_sets_labels(pos, neg, test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    # Gets the positive samples
    features += sample_handling(pos, lexicon, [1,0])
    # Gets the negative samples
    features += sample_handling(neg, lexicon, [0,1])
    random.shuffle(features)

    # Convert the features list to a numpy array
    features = np.array(features)
    # Get the testing size to divide into train and test
    testing_size = int(test_size * len(features))

    # Create training and testing sets
    train_X = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_X = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_X, train_y, test_X, test_y

if __name__== '__main__':
    train_X, train_y, test_X, test_y = create_feature_sets_labels('/home/hduser/neural-networks/sentiment/pos.txt', '/home/hduser/neural-networks/sentiment/neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_X, train_y, test_X, test_y], f)







