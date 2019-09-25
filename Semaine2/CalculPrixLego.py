# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

LEGO_PATH = "U:\IA\IA\Semaine2"

def load_lego_data(lego_path,csvGived):
    csv_path = os.path.join(lego_path,csvGived)
    return pd.read_csv(csv_path)

lego_data_training_set = load_lego_data(LEGO_PATH,"train.csv")
lego_data_training_set.info()

"""lego_data_test_set = load_lego_data(LEGO_PATH,"test.csv")
lego_data_test_set.info()"""

lego_data_training_set.hist(bins=50, figsize=(15,10))
plt.show()




lego_data_training_set = lego_data_training_set.drop(columns=['prod_long_desc'])
lego_data_training_set = lego_data_training_set.drop(columns=['prod_desc'])
lego_data_training_set = lego_data_training_set.drop(columns=['star_rating'])
lego_data_training_set = lego_data_training_set.drop(columns=['val_star_rating'])
lego_data_training_set = lego_data_training_set.drop(columns=['play_star_rating'])
lego_data_training_set = lego_data_training_set.drop(columns=['Unnamed: 0'])
lego_data_training_set = lego_data_training_set.drop(columns=['set_name'])
lego_data_training_set.info()
tableauDesCorrelations = lego_data_training_set.corr()

lego_data_training_set["review_difficulty"].fillna("Unknown", inplace=True)
lego_data_training_set["theme_name"].fillna("Unknown theme_name", inplace=True)


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
review_difficulty = lego_data_training_set["review_difficulty"]
lego_data_training_set["review_difficulty"] = encoder.fit_transform(review_difficulty)

lego_data_training_set["country"] = encoder.fit_transform(lego_data_training_set["country"])
lego_data_training_set["ages"] = encoder.fit_transform(lego_data_training_set["ages"])
lego_data_training_set["theme_name"] = encoder.fit_transform(lego_data_training_set["theme_name"])

tableauDesCorrelations = lego_data_training_set.corr()
lego_data_training_set.info()