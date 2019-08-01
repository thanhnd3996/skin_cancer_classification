import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

# load feature and label from file
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print('Training and making predictions')
clf = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=1000)
clf.fit(X_train, y_train)

predictions = clf.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
