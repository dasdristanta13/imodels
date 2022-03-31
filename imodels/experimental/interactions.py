import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from imodels.experimental.figs_ensembles import FIGSExtClassifier


class InteractionsClassifier(BaseEstimator):
    """Experimental interactions model
    """

    def __init__(self,
                 max_rules: int = 10,
                 n_iters_teacher: int = 1):
        self.teacher = FIGSExtClassifier(max_rules=max_rules)
        self.student = LogisticRegression(solver="lbfgs")
        self.n_iters_teacher = n_iters_teacher

    def fit(self, X, y, **kwargs):
        # fit teacher
        for iter_teacher in range(self.n_iters_teacher):
            self.teacher.fit(X, y, **kwargs)
            y = self.teacher.predict(X)
            # y = self.teacher.predict_proba(X)[:, 1]  # assumes binary classifier

        # fit student
        self.student.fit(X, y)

    def predict(self, X):
        return self.student.predict(X)

    def predict_proba(self, X):
        return self.student.predict_proba(X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == '__main__':
    np.random.seed(13)
    X, y = datasets.load_breast_cancer(return_X_y=True)  # binary classification
    # X, y = datasets.load_diabetes(return_X_y=True)  # regression
    # X = np.random.randn(500, 10)
    # y = (X[:, 0] > 0).astype(float) + (X[:, 1] > 1).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print('X.shape', X.shape)
    print('ys', np.unique(y_train), '\n\n')

    m = InteractionsClassifier(max_rules=5)
    m.fit(X_train, y_train)
    print('test acc', m.score(X_test, y_test))
