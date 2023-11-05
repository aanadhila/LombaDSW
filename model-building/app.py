import pickle
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

pinguins = pd.read_csv('penguins_cleaned.csv')


# membuat salinan dataframe
df = pinguins.copy()
# target yang akan diprediksi
target = 'species'
# yang akan menjadi bahan prediksi
encode = ['sex', 'island']

# mengencoding data dari kolom 'sex' dan 'island'
for col in encode:
    # membuat data dummynya, misal cowok = 1, cewek = 0, dst
    dummy = pd.get_dummies(df[col], prefix=col)
    # masukin data dummy ke dalam dataframe
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# disini kita mengencode target species
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}


def target_encode(val):
    return target_mapper[val]


df['species'] = df['species'].apply(target_encode)

# membagi x dan y

X = df.drop('species', axis=1)
Y = df['species']

# membangun random forest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# saving the model
pickle.dump(clf, open('penguins.clf.pkl', 'wb'))
