import numpy as np
import nltk as nk
import sklearn as sk
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from nltk.corpus import stopwords



def parse_file_into_array(filename):
    with open(filename) as file:
        return file.read().splitlines()

arr_pos = parse_file_into_array("rt-polarity.pos")
arr_neg = parse_file_into_array("rt-polarity.neg")
arr_tot = arr_pos + arr_neg

def to_unigram(input):
    return sk.feature_extraction.text.CountVectorizer().fit_transform(input)

# lemma, stop and stem
def lemmatize(input, stop_words):
    l = nk.stem.WordNetLemmatizer()
    i = 0
    for example in input:
        tokens = nk.word_tokenize(example)
        for token in tokens:
            filtered_words = []
            if token not in stop_words:
                filtered_words.append(l.lemmatize(token))
            input[i] = ' '.join(filtered_words)
        i += 1


def stem(input, stop_words):
    s = nk.stem.PorterStemmer()
    i = 0
    for example in input:
        tokens = nk.word_tokenize(example)
        for token in tokens:
            filtered_words = []
            if token not in stop_words:
                filtered_words.append(s.stem(token))
            input[i] = ' '.join(filtered_words)

        i += 1


def remove_stop_words(input, stop_words):
    i=0
    for example in input:
        tokens = nk.word_tokenize(example)
        for token in tokens:
            filtered_words = []
            if token not in stop_words:
                filtered_words.append(token)
            input[i] = ' '.join(filtered_words)
        i += 1

# # 2 remove infrequent words
# def remove_infrequent_words(input):
#     i=0
#     for example in input:
#         tokens = nk.word_tokenize(example)
#         for token in tokens:
#             filtered_words = []
#
#             input[i] = ' '.join(filtered_words)
#         i += 1


#stop_words is either  or set(nk.corpus.stopwords.words('english')) or set(stopwords.words('english')) since other might cause an error
def preprocess(input, stop_words, s, l):
    if len(stop_words) != 0:
        remove_stop_words(input, stop_words)
    if s:
        stem(input, stop_words)
    if l:
        lemmatize(input, stop_words)


preprocess(arr_tot, [], False, False)

y_pos = [1] * len(arr_pos)
y_neg = [0] * len(arr_neg)

#split data set

training_input, validation_input, training_output, validation_output = \
    sk.model_selection.train_test_split(to_unigram(arr_tot), (y_pos + y_neg), test_size = 0.25, random_state = 35)

def LR(training_input, training_output, validation_input):
    model = sk.linear_model.LogisticRegression(solver = 'liblinear' , random_state = 35)
    model.fit(training_input, training_output)
    return model.predict(validation_input)

def SVM(training_input, training_output, validation_input):
    model = sk.svm.LinearSVC(random_state = 35)
    model.fit(training_input, training_output)
    return model.predict(validation_input)

def NB(training_input, training_output, validation_input):
    model = MultinomialNB(alpha = 0.85)
    model.fit(training_input, training_output)
    return model.predict(validation_input)


def equal_prob(training_input, training_output, validation_input):
    model = DummyClassifier(strategy = 'uniform', random_state= 35)
    model.fit(training_input, training_output)
    return model.predict(validation_input)


results = [LR(training_input, training_output, validation_input), SVM(training_input, training_output, validation_input), \
           NB(training_input, training_output, validation_input), equal_prob(training_input, training_output, validation_input)]

#accuracy of each model
accuracies = []
for model in results:
    accuracies.append(sk.metrics.accuracy_score(validation_output, model))
print(accuracies)

#confusion matrix for best model (Naive_bayes)
cm = sk.metrics.confusion_matrix(validation_output, results[2])
print(cm)
