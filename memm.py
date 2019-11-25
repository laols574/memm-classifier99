"""
training a logistic regression that makes a prediction for each word 
sklearn.feature_extraction.DictVectorizer
sklearn.preprocessing.LabelEncoder
sklearn.linear_model.LogisticRegression
scipy.sparse.vstack
numpy.argmax
"""
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import unicodedata
import re 
import test_memm as tst

from typing import Iterator, Sequence, Text, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix

NDArray = Union[np.ndarray, spmatrix]
TokenSeq = Sequence[Text]
PosSeq = Sequence[Text]

#def main():
    #tst.test_predict_greedy()
    
def read_ptbtagged(ptbtagged_path: str) -> Iterator[Tuple[TokenSeq, PosSeq]]:
    """Reads sentences from a Penn TreeBank .tagged file.
    Each sentence is a sequence of tokens and part-of-speech tags.

    Penn TreeBank .tagged files contain one token per line, with an empty line
    marking the end of each sentence. Each line is composed of a token, a tab
    character, and a part-of-speech tag. Here is an example:

        What	WP
        's	VBZ
        next	JJ
        ?	.

        Slides	NNS
        to	TO
        illustrate	VB
        Shostakovich	NNP
        quartets	NNS
        ?	.

    :param ptbtagged_path: The path of a Penn TreeBank .tagged file, formatted
    as above.
    :return: An iterator over sentences, where each sentence is a tuple of
    a sequence of tokens and a corresponding sequence of part-of-speech tags.
    """
    #do this immediately (first)
    #start generating feature matrices
    
        #read file into an array 
    with open(ptbtagged_path) as f:
        file_array = f.readlines()
    file_array.append("\n")
    array_of_tuples = create_tuples(file_array)

    return generator(array_of_tuples)

def create_tuples(file_array):
    sentence = []
    tags = []
    array_of_tuples = []
    for line in file_array:
        if(line == "\n" or line == "\t\n"):
            array_of_tuples.append((sentence, tags))
            sentence = []
            tags = []
        else:
            line = line.split(None, 1)
            sentence.append(line[0])
            tags.append(line[1].strip())
    return array_of_tuples

def generator(array_of_tuples):
    for i in range(0, len(array_of_tuples)):
        yield array_of_tuples[i]    


            
class Classifier(object):
    def __init__(self):
        """Initializes the classifier."""
        self.clf = LogisticRegression(solver='newton-cg', multi_class="multinomial", dual=False, max_iter=4000) 
        self.features = []
        self.pos_tags = []
        self.l = {}
        self.vectorizer = DictVectorizer()
        self.le = LabelEncoder()
        self.feature_matrix = []
        self.label_vector = []
        self.feature_dict = {}

    def train(self, tagged_sentences: Iterator[Tuple[TokenSeq, PosSeq]]) -> Tuple[NDArray, NDArray]:
        """Trains the classifier on the part-of-speech tagged sentences,
        and returns the feature matrix and label vector on which it was trained.

        The feature matrix should have one row per training token. The number
        of columns is up to the implementation, but there must at least be 1
        feature for each token, named "token=T", where "T" is the token string,
        and one feature for the part-of-speech tag of the preceding token,
        named "pos-1=P", where "P" is the part-of-speech tag string, or "<s>" if
        the token was the first in the sentence. For example, if the input is:

            What	WP
            's	VBZ
            next	JJ
            ?	.

        Then the first row in the feature matrix should have features for
        "token=What" and "pos-1=<s>", the second row in the feature matrix
        should have features for "token='s" and "pos-1=WP", etc. The alignment
        between these feature names and the integer columns of the feature
        matrix is given by the `feature_index` method below.

        The label vector should have one entry per training token, and each
        entry should be an integer. The alignment between part-of-speech tag
        strings and the integers in the label vector is given by the
        `label_index` method below.

        :param tagged_sentences: An iterator over sentences, where each sentence
        is a tuple of a sequence of tokens and a corresponding sequence of
        part-of-speech tags.
        :return: A tuple of (feature-matrix, label-vector).
        """
        #add tokens
        for sentence in tagged_sentences:
            tokens, pos_tags = sentence
            for pos in pos_tags:
                self.pos_tags.append(pos)
            pos_tags.insert(0, "<s>")
            pos_tags.pop(len(pos_tags) - 1)
            for i in range(0, len(tokens)):
                temp_dict = {}
                temp_dict = add_features(tokens,pos_tags[i],i, temp_dict)
                self.features.append(temp_dict)
        #print(self.features)
        feature_matrix = self.vectorizer.fit_transform(self.features)
        label_vector = self.le.fit_transform(self.pos_tags)
        for i in range(0, len(label_vector)):
            self.l[self.pos_tags[i]] = i
        
        self.feature_matrix = feature_matrix
        self.label_vector = label_vector
        self.clf.fit(self.feature_matrix, self.label_vector)

        return (self.feature_matrix, label_vector)


                        
    def feature_index(self, feature: Text) -> int:
        """Returns the column index corresponding to the given named feature.

        The `train` method should always be called before this method is called.

        :param feature: The string name of a feature.
 s       :return: The column index of the feature in the feature matrix returned
        by the `train` method.
        """
        count = 0
        for feature_name in self.vectorizer.get_feature_names():
            if(feature == feature_name):
                return count
            count += 1
    
    def label_index(self, label: Text) -> int:
        """Returns the integer corresponding to the given part-of-speech tag

        The `train` method should always be called before this method is called.

        :param label: The part-of-speech tag string.
        :return: The integer for the part-of-speech tag, to be used in the label
        vector returned by the `train` method.
        """
        count = 0
        for l in self.le.classes_:
            if(l == label):
                return count
            count += 1

    def predict(self, tokens: TokenSeq) -> PosSeq:
        """Predicts part-of-speech tags for the sequence of tokens.

        This method delegates to either `predict_greedy` or `predict_viterbi`.
        The implementer may decide which one to delegate to.

        :param tokens: A sequence of tokens representing a sentence.
        :return: A sequence of part-of-speech tags, one for each token.
        """
        _, pos_tags = self.predict_greedy(tokens)
        # _, _, pos_tags = self.predict_viterbi(tokens)
        return pos_tags

    def predict_greedy(self, tokens: TokenSeq) -> Tuple[NDArray, PosSeq]:
        """Predicts part-of-speech tags for the sequence of tokens using a
        greedy algorithm, and returns the feature matrix and predicted tags.

        Each part-of-speech tag is predicted one at a time, and each prediction
        is considered a hard decision, that is, when predicting the
        part-of-speech tag for token i, the model will assume that its
        prediction for token i-1 is correct and unchangeable.

        The feature matrix should have one row per input token, and be formatted
        in the same way as the feature matrix in `train`.

        :param tokens: A sequence of tokens representing a sentence.
        :return: The feature matrix and the sequence of predicted part-of-speech
        tags (one for each input token).
        """
        
        #array to hold predictions
        predictions = np.zeros((len(tokens), len(self.l), len(self.l)))
        for i in range(len(tokens)):

            for prev_tag in self.l:
                new_feature_matrix = []
                temp_dict = {}
                feature_dict = add_features(tokens, prev_tag ,i, temp_dict)
                new_feature_matrix.append(feature_dict)
                new_feature_matrix = self.vectorizer.transform(new_feature_matrix)
                
                probabilities = self.clf.predict_proba(new_feature_matrix)
                predictions[i, self.label_index(prev_tag)] = probabilities
        
        cur =  len(probabilities[0])- 1
        final_predictions = []
        for i in range(len(predictions)):
            cur_pred = np.argmax(predictions[i, cur])
            final_predictions.append(cur_pred)
            cur = cur_pred
        ret_matrix = []
        #print(tokens)
        final_predictions = self.le.inverse_transform(final_predictions)
        #print(final_predictions)
        new_pos = final_predictions.tolist()
        new_pos.insert(0, "<s>")
        new_pos.pop(len(new_pos) - 1)
        for i in range(len(new_pos)):
            feature_dict = add_features(tokens, new_pos[i] ,i, temp_dict)
            ret_matrix.append(feature_dict)
        ret_matrix = self.vectorizer.transform(ret_matrix)

        return (ret_matrix, final_predictions)


    def predict_viterbi(self, tokens: TokenSeq) -> Tuple[NDArray, NDArray, PosSeq]:
        """Predicts part-of-speech tags for the sequence of tokens using the
        Viterbi algorithm, and returns the transition probability tensor,
        the Viterbi lattice, and the predicted tags.

        The entry i,j,k in the transition probability tensor should correspond
        to the log-probability estimated by the classifier of token i having
        part-of-speech tag k, given that the previous part-of-speech tag was j.
        Thus, the first dimension should match the number of tokens, the second
        dimension should be one more than the number of part of speech tags (the
        last entry in this dimension corresponds to "<s>"), and the third
        dimension should match the number of part-of-speech tags.

        The entry i,k in the Viterbi lattice should correspond to the maximum
        log-probability achievable via any path from token 0 to token i and
        ending at assigning token i the part-of-speech tag k.

        The predicted part-of-speech tags should correspond to the highest
        probability path through the lattice.

        :param tokens: A sequence of tokens representing a sentence.
        :return: The transition probability tensor, the Viterbi lattice, and the
        sequence of predicted part-of-speech tags (one for each input token).
        """
#pick best path through POS,
def add_features(tokens, pos, i, temp_dict):
    #intialize
    temp_dict = {"token" : tokens[i] , "pos-1" : pos}

    #word shape
    temp_dict["word_shape"] = shape(tokens[i])
    #temp_dict["ion"] = ion(tokens[i])
    temp_dict["ing"] = ing(tokens[i])
    temp_dict["able"] = able(tokens[i])
    temp_dict["ly"] = ly(tokens[i])
    temp_dict["ed"] = ed(tokens[i])
    #temp_dict["ness"] = ness(tokens[i])
    #temp_dict["al"] = al(tokens[i])
    #temp_dict["ive"] = ive(tokens[i])
    temp_dict["s"] = s(tokens[i])
    #temp_dict["ity"] = ity(tokens[i])
    temp_dict["capital"] = capital(tokens[i])
    temp_dict["punct"] = punct(tokens[i])
    #first word
    
    temp_dict["first_word"] = int(i == 0)

    #last word
    temp_dict["last_word"] = int(i == len(tokens) - 1)
    
    #all caps
    temp_dict['is_complete_capital'] = int(tokens[i].upper()==tokens[i])
    
    #prev_word
    if i != 0:
        temp_dict['prev_word'] =  tokens[i-1]
        
        
    #next word
    if i < len(tokens) - 1:
        temp_dict['next_word'] = tokens[i+1]
        
        
    #is numeric
    temp_dict['is_numeric'] = int(tokens[i].isdigit())
    temp_dict['is_alphanumeric'] =  int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',tokens[i])))),
    temp_dict['prefix_1'] = tokens[i][0]
    temp_dict['prefix_2']= tokens[i][:2]
    temp_dict['prefix_3'] = tokens[i][:3]
    temp_dict['prefix_4'] = tokens[i][:4]
    temp_dict['suffix_1'] = tokens[i][-1]
    temp_dict['suffix_2'] = tokens[i][-2:]
    temp_dict['suffix_3'] = tokens[i][-3:]
    temp_dict['suffix_4'] = tokens[i][-4:]
    
    if "-" in tokens[i]:
        temp_dict['word_has_hyphen'] = 1
    else:
        temp_dict['word_has_hyphen'] = 0
    #n grams
    
    if(i < len(tokens) - 1):
        temp_dict["bigrams"] = bigrams(tokens[i], tokens[i + 1])

    return temp_dict


#ion
def punct(word):
    punct = re.compile('[!$%?.,]')
    match = punct.match(word)
    if(match != None):
        return 1
    return 0

#ion
def ion(word):
    return int(word.endswith("ion"))

#ly
def ly(word):
    return int(word.endswith("ly"))
#ed
def ed(word):
    return int(word.endswith("ed"))

#ing
def ing(word):
    return int(word.endswith("ing"))

#able
def able(word):
    return int(word.endswith("able"))

#ness
def ness(word):
    return int(word.endswith("ness"))

#ity
def ity(word):
    return int(word.endswith("ity"))

#ity
def ive(word):
    return int(word.endswith("ive"))

#ity
def al(word):
    return int(word.endswith("al"))
#ity
def s(word):
    return int(word.endswith("s"))

#capital
def capital(word):
    return int(word[0].isupper())

def shape(word):
    f = unicodedata.category
    return ''.join(map(f, word))

def bigrams(word1, word2):
    return word1 + "_" + word2

def trigrams(word1, word2, word3):
    return word1 + "_" + word2 + "_" + word3


#if __name__ == "__main__":
   # main()