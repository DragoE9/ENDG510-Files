import numpy
import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

data = pandas.read_csv("data.csv")
data = data.drop_duplicates()

y = data['Label']
x = data.drop(['Label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = GaussianNB()
model.fit(x_train,y_train)

probs_train = model.predict(x_train)
table = confusion_matrix(y_train, probs_train)
accuracy = accuracy_score(y_train, probs_train)
print("Training Set:\n",table,f"\nAccuracy:{accuracy:.4f}\n")

probs_test = model.predict(x_test)
table = confusion_matrix(y_test, probs_test)
accuracy = accuracy_score(y_test, probs_test)
print("Test Set:\n",table,f"\nAccuracy:{accuracy:.4f}\n")

pickle.dump(model, open('SensorValidator.pkl', 'wb'))