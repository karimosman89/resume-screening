import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

def train_model(features, labels):
    """
    Train a Multinomial Naive Bayes classifier on the processed resume data.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:", report)
    
    return model

def save_model(model, file_path="model.pkl"):
    """
    Save the trained model to a file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path="model.pkl"):
    """
    Load a saved model from file.
    """
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on the test set.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

