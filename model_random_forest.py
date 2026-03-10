from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from utils import print_confusion_matrix

THRESHOLD = 0.6
N_ESTIMATORS = 300
MAX_DEPTH = 6

class RandomForest:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > THRESHOLD).astype(int)

        print_confusion_matrix(y_test, y_pred, title='Confusion Matrix for Random Forest Model in Incident Detection')
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        return accuracy, recall, precision