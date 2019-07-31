import xgboost as xgb
from sklearn.metrics import accuracy_score

# load feature and label from file
X_train = [line.strip() for line in open('X_train.txt')]
y_train = [line.strip() for line in open('y_train.txt')]
X_val = [line.strip() for line in open('X_val.txt')]
y_val = [line.strip() for line in open('y_val.txt')]

print('Training and making predictions')
clf = xgb.XGBClassifier(max_depth=15, learning_rate=0.1, n_estimators=1000)
clf.fit(X_train, y_train)

predictions = clf.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100))
