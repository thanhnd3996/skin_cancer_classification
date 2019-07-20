import pandas as pd
from sklearn.model_selection import train_test_split

train_labels = pd.read_csv("ISIC2018_Task3_Training_GroundTruth.csv")


# print(train_labels.head())


def diagnosis(row):
    return row[row == 1].index[0]


train_labels['diagnosis'] = train_labels.apply(diagnosis, axis=1)
print(train_labels.shape)
train_labels.drop(columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], inplace=True)

train, val = train_test_split(train_labels, test_size=0.2)
print(train.groupby('diagnosis').count)
print(val.groupby('diagnosis').count)
# print(train)
# print(test)

train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)