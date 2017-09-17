import numpy as np
from os import listdir, path
import tarfile

def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)


def import_data(path_to_root):
    if "data" not in listdir(path.join(path_to_root, "notebooks")):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open(path.join(path_to_root, "data.tar.gz"))
        tarObject.extractall()
        tarObject.close()
        print("Extracted data.tar.gz to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY


def import_smsspam(path_to_root):
    if "smsspam" not in listdir(path.join(path_to_root, "notebooks")):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open(path.join(path_to_root, "spmsspam.tar.gz"))
        tarObject.extractall()
        tarObject.close()
        print("Extracted smsspam.tar.gz to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data; 1 == spam, 2 == ham")
    train_docs, train_labels = parse_smsspam_data(
        path.join(path_to_root, "notebooks/smsspam/SMSSpamCollection.train")
    )

    print("loading dev data; 1 == spam, 2 == ham")
    dev_docs, dev_labels = parse_smsspam_data(
        path.join(path_to_root, "notebooks/smsspam/SMSSpamCollection.devel")
    )

    print("loading testing data; 1 == spam, 2 == ham")
    test_docs, test_labels = parse_smsspam_data(
        path.join(path_to_root, "notebooks/smsspam/SMSSpamCollection.test")
    )

    return train_docs, train_labels, dev_docs, dev_labels, test_docs, test_labels


def parse_smsspam_data(path_to_smsspam_file):
    tokenized_docs = []
    labels = []
    with open(path_to_smsspam_file, "r") as f:
        for line in f:
            # split line
            raw_label, raw_doc = line.split("\t")
            # convert raw label
            label = 1 if raw_label == "spam" else 0
            # whitespace tokenize doc
            tokenized_doc = raw_doc.split()
            tokenized_docs.append(tokenized_doc)
            labels.append(label)
    return tokenized_docs, labels


def build_w2i_lookup(training_corpus):
    lookup = {"<unk>": 0}
    c = 1
    for doc in training_corpus:
        for word in doc:
            word = word.lower()
            if word not in lookup:
                c += 1
                lookup[word] = c
    return lookup


def load_pretrained_embeddings(path_to_file, take=None):
    embedding_size = None
    embedding_matrix = None
    lookup = {"<unk>": 0}
    c = 0
    with open(path_to_file, "r") as f:
        for line in f:
            if c == 0:
                # check for header line
                if len(line.split()) == 2:
                    c = 1
                    pass
            else:
                # check for delimiter
                if "\t" in line:
                    delimiter = "\t"
                else:
                    delimiter = " "
                if take and c <= take:
                    # split line
                    line_split = line.rstrip().split(delimiter)
                    # extract word and vector
                    word = line_split[0]
                    vector = np.array([float(i) for i in line_split[1:]])
                    # get dimension of vector
                    embedding_size = vector.shape[0]
                    # add to lookup
                    lookup[word] = c
                    # add to embedding matrix
                    if np.any(embedding_matrix):
                        embedding_matrix = np.vstack((embedding_matrix, vector))
                    else:
                        embedding_matrix = np.zeros((2, embedding_size))
                        embedding_matrix[1] = vector
                    c += 1
    return embedding_matrix, lookup
