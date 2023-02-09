# Assignment5
## . 1) Implement Naïve Bayes method using scikit-learn library
     

glass_data = pd.read_csv("glass.csv")
X = glass_data.iloc[:, :-1]
y = glass_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




### 2)Implement linear SVM method using scikit library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
glass = datasets.load_glass()
X = glass.data
y = glass.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

### The accuracy of the model depends on the specific data, modal parameters, and other factors. It's not possible to determine which algorithm will have better accuracy without evaluating it on the specific data. To justify the choice of algoritm, one would need to compare the accuracy, recall, and other metrics for each algorithm and select the one that perfoems best.
