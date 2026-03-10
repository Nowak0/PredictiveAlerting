from sklearn.metrics import accuracy_score, recall_score, precision_score
import xgboost as xgb
from utils import print_confusion_matrix


THRESHOLD = 0.6
N_ESTIMATORS = 300
MAX_DEPTH = 6


class XGBoost:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        train_data = xgb.DMatrix(X_train, y_train)
        amount_no_incident = y_train[y_train == 0].shape[0]
        amount_incident = y_train[y_train == 1].shape[0]
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'scale_pos_weight': amount_no_incident / amount_incident,
            'max_depth': MAX_DEPTH
        }

        self.model = xgb.train(params, train_data, num_boost_round=N_ESTIMATORS)

    def evaluate(self, X_test, y_test):
        test_data = xgb.DMatrix(X_test, y_test)

        y_prob = self.model.predict(test_data)
        y_pred = (y_prob > THRESHOLD).astype(int)

        print_confusion_matrix(y_test, y_pred, title='Confusion Matrix for XGBoost Model in Incident Detection')
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        return accuracy, recall, precision