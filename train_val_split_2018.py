import pandas as pd
from sklearn.model_selection import train_test_split

train_labels = pd.read_csv("ISIC2018_Task3_Training_GroundTruth.csv")


# print(train_labels.head())


def diagnosis(row):
    return row[row == 1].index[0]


train_labels['diagnosis'] = train_labels.apply(diagnosis, axis=1)
train_labels.drop(columns=['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'], inplace=True)
# print(train_labels.head())
train_labels.to_csv("train_val_2018.csv")
