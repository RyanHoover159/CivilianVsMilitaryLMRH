from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

def train_models(features, labels):

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)

    rf.fit(X_train, y_train)
    oc.fit(X_train, y_train)

    return {"Random Forest": (rf, X_test, y_test), "One-Class SVM": (oc, X_test, y_test)}