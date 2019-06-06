import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.classification import  accuracy_score
from sklearn.metrics.classification import  log_loss
import sklearn
import matplotlib.pyplot as plt

np.random.seed(1)

from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    diabetes = datasets.fetch_openml('diabetes')
    y = sklearn.preprocessing.LabelEncoder().fit_transform(diabetes['target'])

    X_train, X_test, y_train, y_test = train_test_split(diabetes['data'], y)

    preds = []

    accs = []

    for x in range(0, 500):
        model = DecisionTreeClassifier()

        X_train_cur, _, y_train_cur, _ = train_test_split(X_train, y_train)

        model.fit(X_train_cur, y_train_cur)

        y_hat = model.predict_proba(X_test)

        preds.append(y_hat)

        acc = accuracy_score(np.argmax(np.mean(preds, axis=0), axis=1), y_test)
        accs.append(acc)
        print(acc)

    plt.plot(np.arange(1, 501), accs, label='Accuracy')
    plt.savefig('../figures/ensemble.pdf')


if __name__ == "__main__":
    main()
