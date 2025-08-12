from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_FILE = os.path.join(os.path.dirname(__file__), "../model/no_show_model.pkl")

# Load model and feature names
model, feature_names = joblib.load(MODEL_FILE)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None
    form_data = {}
    if request.method == "POST":
        try:
            form_data['Age'] = int(request.form.get('Age', 35))
            form_data['Gender'] = int(request.form.get('Gender', 1))
            form_data['BookingLeadTime'] = int(request.form.get('BookingLeadTime', 5))
            form_data['PreviousNoShows'] = int(request.form.get('PreviousNoShows', 0))
            form_data['SMSReminderSent'] = int(request.form.get('SMSReminderSent', 1))
            form_data['ChronicConditions'] = int(request.form.get('ChronicConditions', 0))
            form_data['DistanceToClinic'] = float(request.form.get('DistanceToClinic', 3.2))
        except ValueError:
            result = "Please enter valid numeric values."
            return render_template("index.html", result=result, form=form_data)

        # Prepare input data frame for prediction
        X_new = pd.DataFrame([form_data], columns=feature_names)
        prediction = model.predict(X_new)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_new)[0][1]
        else:
            prob = None

        result = "No-Show" if prediction == 1 else "Will Show"

    return render_template("index.html", result=result, prob=prob, form=form_data)

if __name__ == "__main__":
    app.run(debug=True)
