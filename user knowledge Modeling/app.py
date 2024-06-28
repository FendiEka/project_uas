from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model_filename = "models/knowledge_classifier.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Load data untuk referensi
data = pd.read_csv('data_Knowledge.csv')

# Pastikan kolom target adalah 'target'
X = data[['STG', 'SCG', 'STR', 'LPR', 'PEG']]
y = data['target']

# Class names
class_names = {
    0: "very_low",
    1: "low",
    2: "middle",
    3: "high"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Mengambil nilai input dari form dan melakukan validasi tipe data
        try:
            input_data = {
                "STG": float(request.form["STG"]),
                "SCG": float(request.form["SCG"]),
                "STR": float(request.form["STR"]),
                "LPR": float(request.form["LPR"]),
                "PEG": float(request.form["PEG"])
            }
        except ValueError:
            return render_template("index.html", prediction="Invalid input. Please enter numeric values only.", form_data=None)

        # Convert ke DataFrame yang sesuai dengan struktur training data
        mydata = pd.DataFrame([input_data], columns=['STG', 'SCG', 'STR', 'LPR', 'PEG'])

        # Prediksi kelas
        try:
            predictions = model.predict(mydata)
            predicted_class = predictions[0]
            predicted_class_name = class_names.get(predicted_class, f"Unknown (class {predicted_class})")
        except Exception as e:
            return render_template("index.html", prediction=f"Error in prediction: {str(e)}", form_data=input_data)

        return render_template("index.html", prediction=predicted_class_name, form_data=input_data)

    return render_template("index.html", prediction=None, form_data=None)

if __name__ == "__main__":
    app.run(debug=True)
