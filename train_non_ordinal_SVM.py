import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm

def get_data(shuffle_seed = 0):
    data = np.load(open('non_ordinal_dataset.npy', 'rb')).astype(np.float32)
    np.random.seed(shuffle_seed)
    np.random.shuffle(data)
    features = data[:, :-1]
    labels = data[:,-1]
    return features, labels

if __name__ == "__main__":
    features, labels = get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("Precision:",metrics.precision_score(y_test, y_pred))
    print("Recall:",metrics.recall_score(y_test, y_pred))
    