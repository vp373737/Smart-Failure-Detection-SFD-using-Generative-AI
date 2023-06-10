import pandas as pd
import pickle
import time
import numpy as np
import serial
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ser = serial.Serial('COM6', 9600)
ser.flushInput()
fields = ['Time','Sound','Vibration','Temperature','Humidity']
f = open("../Data Collection/TTrainingData.csv", "a+")
writer = csv.writer(f, delimiter=',')
writer.writerow(fields)

# Load data
data = pd.read_csv('../Data Collection/TrainingData.csv')

# Preprocess data
data.dropna(inplace=True)
#data['is_maintenance'] = data['Maintenance'].notnull().astype(int)
X = data[['Sound', 'Temperature', 'Vibration', 'Humidity']]
y = data['Maintenance']



# Generate synthetic data for good scenarios using generative AI techniques
def generate_synthetic_goodData(num_samples=1000):
    synthetic_data = pd.DataFrame()
    for _ in range(num_samples):
        # Generate synthetic data by randomizing the features within a certain range
        synthetic_instance = {
            'Sound': np.random.uniform(low=30, high=1007),
            'Vibration': np.random.uniform(low=100, high=1023),
            'Temperature': np.random.uniform(low=28, high=50),
            'Humidity': np.random.uniform(low=20, high=31),
            'Maintenance': 1  # Assign label 1 to synthetic data
        }
        synthetic_data = pd.concat([synthetic_data, pd.DataFrame(synthetic_instance, index=[0])], ignore_index=True)
    return synthetic_data

# Generate synthetic data for failure scenarios using generative AI techniques
def generate_synthetic_badData(num_samples=2000):
    synthetic_data = pd.DataFrame()
    for _ in range(num_samples):
        # Generate synthetic data by randomizing the features within a certain range
        synthetic_instance = {
            'Sound': np.random.uniform(low=1008, high=2000),
            'Vibration': np.random.uniform(low=1024, high=2000),
            'Temperature': np.random.uniform(low=51, high=100),
            'Humidity': np.random.uniform(low=32, high=50),
            'Maintenance': 0  # Assign label 0 to synthetic data
        }
        synthetic_data = pd.concat([synthetic_data, pd.DataFrame(synthetic_instance, index=[0])], ignore_index=True)
    return synthetic_data

synthetic_goodData = generate_synthetic_goodData()

# Combine real and synthetic data
combined_goodFeatures = pd.concat([X, synthetic_goodData[['Sound', 'Vibration', 'Temperature', 'Humidity']]])
combined_goodLabels = pd.concat([y, synthetic_goodData['Maintenance']])

synthetic_badData = generate_synthetic_badData()

# Combine real and synthetic data
combined_Features = pd.concat([combined_goodFeatures, synthetic_badData[['Sound', 'Vibration', 'Temperature', 'Humidity']]])
combined_Labels = pd.concat([combined_goodLabels, synthetic_badData['Maintenance']])

X_train, X_test, y_train, y_test = train_test_split(combined_Features, combined_Labels, test_size=0.2, random_state=42)

start= time.time()
print("Start Time- in secs" ,(start))
# Train a KNN classifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Save the trained model to a pickle file
with open('./Models/KNNmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)
stop= time.time()
print("Stop Time- in secs" ,(stop))
print("Elapse Time- in secs" ,(stop-start))

start= time.time()
print("Start Time- in secs" ,(start))
# Train a logistic regression classifier
clf = LogisticRegression(random_state=4)
clf.fit(X_train, y_train)

# Make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Save the trained model to a pickle file
with open('./Models/LRmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)
stop= time.time()
print("Stop Time- in secs" ,(stop))
print("Elapse Time- in secs" ,(stop-start))

start= time.time()
print("Start Time- in secs" ,(start))
# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Save the trained model to a pickle file
with open('./Models/DTmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)
stop= time.time()
print("Stop Time- in secs" ,(stop))
print("Elapse Time- in secs" ,(stop-start))

start= time.time()
print("Start Time- in secs" ,(start))
# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
#print(X_test)
print('Confusion matrix:\n', confusion_mat)
# Save the trained model to a pickle file
with open('./Models/RFmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)
stop= time.time()
print("Stop Time- in secs" ,(stop))
print("Elapse Time- in secs" ,(stop-start))

start= time.time()
print("Start Time- in secs" ,(start))
# Train a SVM classifier
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test data and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
# Save the trained model to a pickle file
with open('./Models/SVMmodel.pkl', 'wb') as f:
    pickle.dump(clf, f)
stop= time.time()
print("Stop Time- in secs" ,(stop))
print("Elapse Time- in secs" ,(stop-start))