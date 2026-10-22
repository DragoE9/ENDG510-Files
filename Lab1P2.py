import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
data = pandas.read_csv("data.csv")

#Check the loaded data
print(data.head())

#Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n",missing_values)

#De-Duplicate the data
data = data.drop_duplicates()

#Prepare features
y = data['Label']
x = data.drop(['Label'], axis=1)

#Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Data Standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Models
models = {
	'KNN': KNeighborsClassifier(n_neighbors=5),
	'SVM': SVC(kernel='rbf'),
	'Random Forest': RandomForestClassifier(n_estimators=10),
	'Naive Bayes': GaussianNB(),
	'QDA': QuadraticDiscriminantAnalysis(store_covariance=True)
}

#train and evaluate models
for name, model in models.items():
	scores = cross_val_score(model, x_train, y_train, cv=5)
	print(f"{name:20s} {scores.mean():.3f} (+/- {scores.std():.3f})")