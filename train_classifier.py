#! C:\Users\Asus\Python\mvenv\Scripts\python.exe



# In this file to train the data, to take the data, to load the data

import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data_dict = pickle.load(open('./data.pickle', 'rb'))             #to load the data from the pickle file

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])



#Add Filtering mask


unique_labels, counts = np.unique(labels, return_counts=True)                   # 1. Find the unique labels and how many times they appear

labels_to_keep = unique_labels[counts > 1]                                          # 2. Find the labels that appear more than once (at least 2 times)


mask = np.isin(labels, labels_to_keep)                                      # 3. Create a "mask" to filter your data and labels


# 4. Apply the mask
data_filtered = data[mask]
labels_filtered = labels[mask]

print(f"Original dataset size: {len(labels)}")
print(f"Filtered dataset size: {len(labels_filtered)}")



# Now, use the FILTERED data for splitting
x_train, x_test, y_train, y_test = train_test_split(data_filtered, labels_filtered, 
                                                    test_size=0.3, 
                                                    shuffle=True, 
                                                    stratify=labels_filtered) # Use filtered labels for stratify

model = RandomForestClassifier()                     # To fit the training data into the model
model.fit(x_train, y_train)     

y_predict = model.predict(x_test)                    # To predict the testing data

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()