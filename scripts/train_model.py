# scripts/train_model.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

os.makedirs("../model", exist_ok=True)
DATA_PATH = "../data/appointments_1m.csv"
MODEL_PATH = "../model/no_show_model.pkl"

def load_data(n_rows=None):
    print("Loading data...")
    return pd.read_csv(DATA_PATH, nrows=n_rows)

def prepare(df):
    features = ['Age', 'Gender', 'BookingLeadTime', 'PreviousNoShows',
                'SMSReminderSent', 'ChronicConditions', 'DistanceToClinic']
    X = df[features]
    y = df['NoShow']
    return X, y

def train_sgd(save_path=MODEL_PATH, test_size=0.2, n_rows=None):
    df = load_data(n_rows)
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("Training SGDClassifier (logistic regression via SGD)...")
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump((model, X.columns.tolist()), save_path)
    print(f"Saved model to {save_path}")

def train_random_forest(save_path="../model/no_show_model_rf.pkl", subsample=100000):
    # Use a smaller subsample for RF to keep it tractable
    df = load_data(n_rows=subsample)
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training RandomForest on subsample (this may take a while)...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump((model, X.columns.tolist()), save_path)
    print(f"Saved RF model to {save_path}")

if __name__ == "__main__":
    # default: train SGD on full file
    train_sgd(n_rows=None)   # set n_rows to limit memory/time for testing
    # train_random_forest(subsample=100000)  # uncomment to train RF on a subsample
