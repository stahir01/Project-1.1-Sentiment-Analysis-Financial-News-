import string
import inflect
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




def extract_data(file_path) -> (list, list):
    """
    Reads in the data and labels from the provided .txt file.
    The output will contain everything that was in the input file, except the delimiter (@).
    :param file_path:
    :return:
    """
    corpus, labels = [], []
    with open(file_path, 'r',  encoding='latin-1') as f:
        for line in f:
            doc, label = line.split('@')
            corpus.append(doc)
            labels.append(label.rstrip('\n'))
    return corpus, labels


def normalize_corpus(data, remove_numbers=False) -> list:
    """
    Will normalize the data with the following methods:
    - convert everything into lower case
    - replace '%' with the word 'percent'
    - remove punctuation contained in string.punctuation
    - turn numbers into their word equivalent (e.g., 105 -> one hundred and five)
    - tokenize all the remaining words
    :param remove_numbers: Will remove numbers from the sentences, default=False
    :param data: corpus of text
    :return: array of normalized sentences/documents with their word tokens
    """
    lower_data = [line.lower() for line in data]
    replaced_data = [re.sub(r'%', 'percent', line) for line in lower_data]

    # Create translation table without minus sign
    translation_table = str.maketrans("", "", string.punctuation.replace("-", ""))

    if remove_numbers:
        replaced_data = [re.sub(r'\d+', '', line) for line in replaced_data]

    # Note: We keep the - of the range as it contains information
    split_number = [
        re.sub(r'(?<!-)\b(\d+)\s*-\s*(\d+)\b(?!-)', lambda match: f'{match.group(1)} - {match.group(2)}', line)
        for line in replaced_data
    ]

    no_punct_data = [line.translate(translation_table) for line in split_number]
    p = inflect.engine()
    normalized_data = []

    for line in no_punct_data:
        tokens = word_tokenize(line)
        normalized_tokens = []
        for token in tokens:
            if token.isdigit():
                words = p.number_to_words(token).split()
                normalized_tokens.extend(words)
            else:
                normalized_tokens.append(token)
        normalized_data.append(normalized_tokens)

    return normalized_data


def get_data_splitted(norm_data, labels):
    positive, neutral, negative = [], [], []
    for sentence, label in zip(norm_data, labels):
        if label == 'positive':
            positive.append(sentence)
        if label == 'neutral':
            neutral.append(sentence)
        else:
            negative.append(sentence)
    return positive, neutral, negative


def remove_stopwords(data: list) -> list:
    filtered = []
    stop_words = set(stopwords.words('english'))
    for sentence in data:
        filtered.append([])
        for word in sentence:
            if word not in stop_words:
                filtered[-1].append(word)
    return filtered


def stem_and_lemmatize_text(data: list) -> list:
    """
    Stems and lemmatizes the words in the given data.
    :param data: List of lists of words.
    :return: List of lists of stemmed and lemmatized words.
    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_and_lemmatized = []
    
    for sentence in data:
        stemmed_sentence = []
        for word in sentence:
            stemmed_word = lemmatizer.lemmatize(stemmer.stem(word))
            stemmed_sentence.append(stemmed_word)
        stemmed_and_lemmatized.append(stemmed_sentence)
    
    return stemmed_and_lemmatized


def Confusion_Matrix(y_test, y_pred):
    unique_classes = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
    return disp


def get_data_cases(case, cases_ff_net):
    """
    Returns the data for the specified case.
    Args:
        case (int): The case for which the data is to be returned.
        cases_ff_net (dict): The dictionary containing the data for all cases.
    Returns:
        X_train (list): The training data for the specified case.   
        y_train (list): The corresponding labels for the training data.
        X_test (list): The test data for the specified case.
        y_test (list): The corresponding labels for the test data.
        label_encoder (object): The label encoder used to encode the labels.
    """
    return cases_ff_net[case]["x_train"], cases_ff_net[case]["y_train"], cases_ff_net[case]["x_test"], cases_ff_net[case]["y_test"], cases_ff_net[case]["label_encoder"]



# Summarize all preprocessing steps in one function
def data_preprocessing(data, labels, remove_numbers=False, remove_sw=False, stem_and_lemmatize=False):
    """
    Preprocesses the data by normalizing, removing stopwords, and stemming and lemmatizing the words.

    Args:
        data (list): The input data for training.
        labels (list): The corresponding labels for the input data.
        remove_numbers (bool): Whether to remove numbers from the data.
        remove_sw (bool): Whether to remove stopwords from the data.
        stem_and_lemmatize (bool): Whether to stem and lemmatize the words in the data.

    Returns:
        data (list): The preprocessed data.
        labels (list): The corresponding labels for the preprocessed data.
    """

    # Normalize data
    data = normalize_corpus(data, remove_numbers=remove_numbers)

    # Remove stopwords
    if remove_sw:
        data = remove_stopwords(data)

    # Stem and lemmatize words
    if stem_and_lemmatize:
        data = stem_and_lemmatize_text(data)

    data = [' '.join(sentence) for sentence in data]

    return data, labels



def train_classifier(model, vectorizer, data, labels):
    """
    Trains a model using the provided data and labels.

    Args:
        model (object): The machine learning model to be trained.
        vectorizer (object): The feature vectorizer or transformer.
        data (list): The input data for training.
        labels (list): The corresponding labels for the input data.

    Returns:
        report (str): Classification report containing precision, recall, F1-score, and support.
        f1 (float): F1-score of the trained model.
        cm (array): Confusion matrix of the trained model.
    """

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True,random_state=42)

    pipeline = Pipeline([('vect', vectorizer),
                         ('model', model)])
    
    pipeline.fit(X_train, y_train)

    accuracy_score = pipeline.score(X_test, y_test) * 100

    #print(f"Accuracy: {pipeline.score(X_test, y_test) * 100}%")

    y_pred = pipeline.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Calculate confusion matrix
    cm = Confusion_Matrix(y_test, y_pred)

    return accuracy_score, f1, report, cm


# Plot loss and accuracy scores for TensorFlow models
def plot_loss_accuracy(model_results):
    """
    Plots the loss and accuracy scores of a TensorFlow model.
    Args:
        model_results (object): The history object of the trained model.
    
    Returns:
        None
    """
    train_loss = model_results.history['loss']
    val_loss = model_results.history['val_loss']
    train_accuracy = model_results.history['accuracy']
    val_accuracy = model_results.history['val_accuracy']


    epochs = range(1, len(train_loss) + 1)

    # Create subplots with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))

    # Plot loss scores
    axs[0].plot(epochs, train_loss, 'g', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    axs[0].set_title('Loss Scores')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot accuracy scores
    axs[1].plot(epochs, train_accuracy, 'c', label='Training Accuracy')
    axs[1].plot(epochs, val_accuracy, 'm', label='Validation Accuracy')
    axs[1].set_title('Accuracy Scores')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Show the plots
    plt.show()


def report_and_confusion_matrix(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Decode the predicted classes
    predicted_labels = label_encoder.inverse_transform(y_pred_classes)

    # Decode the true classes
    true_labels = label_encoder.inverse_transform(y_test)

    # Generate the classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)

    conf_matrix = Confusion_Matrix(true_labels, predicted_labels)    
    conf_matrix.plot()
    plt.show()


