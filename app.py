from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import os
import time
import io

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_PATH = "server_data_advanced.csv"
SCRIPT_PATH = "train_model.py"

# --- 1. LOAD MAIN ARTIFACTS (For the real Predict Page) ---
try:
    # These are GLOBAL variables for the /predict route only
    main_model = pickle.load(open("model.pkl", "rb"))
    main_scaler = pickle.load(open("scaler.pkl", "rb"))
    main_encoders = pickle.load(open("encoders.pkl", "rb"))
    print("✅ Main System Loaded: Model, Scaler, Encoders ready.")
except:
    print("⚠️ Artifacts not found. Run train_model.py first.")


# --- 2. HELPER FUNCTIONS ---
def process_input(data):
    try:
        input_data = pd.DataFrame(
            [
                {
                    "Provider": data["provider"],
                    "Server_Type": data["server_type"],
                    "Region": data["region"],
                    "OS_Type": data["os_type"],
                    "Hour": int(data["hour"]),
                    "Active_Users": int(data["active_users"]),
                    "Network_Packets_In": int(data["network_packets"]),
                    "Disk_IO_Speed": float(data["disk_io"]),
                    "Memory_Usage_Percent": float(data["memory_usage"]),
                }
            ]
        )

        categorical_cols = ["Provider", "Server_Type", "Region", "OS_Type"]
        for col in categorical_cols:
            le = main_encoders[col]
            if input_data[col][0] not in le.classes_:
                return None, f"Invalid {col}"
            input_data[col] = le.transform(input_data[col])

        return main_scaler.transform(input_data), None
    except Exception as e:
        return None, str(e)


# --- 3. PAGE ROUTES ---
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/training")
def training_page():
    return render_template("training.html")


@app.route("/testing")
def testing_page():
    return render_template("testing.html")


@app.route("/download")
def download_page():
    return render_template("download.html")


# --- 4. SIMULATION LOGIC (Isolated) ---
@app.route("/run_step/<int:step_id>", methods=["POST"])
def run_step(step_id):
    try:
        output = ""

        # NOTE: We re-import libraries inside steps to ensure isolation and robustness

        # Step 1: Import & Load
        if step_id == 1:
            df = pd.read_csv(DATA_PATH)
            output = ">>> import pandas as pd\n"
            output += f">>> df = pd.read_csv('{DATA_PATH}')\n"
            output += ">>> print(df.head())\n\n"
            output += "First 5 Rows:\n" + df.head().to_string()

        # Step 2: Info
        elif step_id == 2:
            df = pd.read_csv(DATA_PATH)
            buffer = io.StringIO()
            df.info(buf=buffer)
            output = ">>> print(df.info())\n\n" + buffer.getvalue()

        # Step 3: Describe
        elif step_id == 3:
            df = pd.read_csv(DATA_PATH)
            output = ">>> print(df.describe())\n\n" + df.describe().to_string()

        # Step 4: Missing Values & Encoders Setup
        elif step_id == 4:
            df = pd.read_csv(DATA_PATH)
            output = ">>> print(df.isnull().sum())\n\n" + df.isnull().sum().to_string()
            output += "\n\n>>> encoders = {}\n"
            output += ">>> categorical_cols = ['Provider', 'Server_Type', 'Region', 'OS_Type']"

        # Step 5: Encoding Loop
        elif step_id == 5:
            from sklearn.preprocessing import LabelEncoder

            df = pd.read_csv(DATA_PATH)
            categorical_cols = ["Provider", "Server_Type", "Region", "OS_Type"]

            output = ">>> for col in categorical_cols:\n"
            output += "...     le = LabelEncoder()\n"
            output += "...     df[col] = le.fit_transform(df[col])\n"
            output += "...     encoders[col] = le\n\n"

            # Actually perform it for display
            output += "Encoding Log:\n"
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                output += f" -> Encoded '{col}'\n"

        # Step 6: Verify Encoding
        elif step_id == 6:
            # Re-run encoding to get state
            from sklearn.preprocessing import LabelEncoder

            df = pd.read_csv(DATA_PATH)
            for col in ["Provider", "Server_Type", "Region", "OS_Type"]:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

            output = ">>> print('Encoded Data Sample')\n"
            output += ">>> print(df.head())\n\n" + df.head().to_string()

        # Step 7: Split
        elif step_id == 7:
            output = ">>> X = df.drop(columns=['CPU_Load'])\n"
            output += ">>> y = df['CPU_Load']\n"
            output += ">>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n"
            output += "Data Split Successfully:\n"
            output += " -> Training Set: 1600 rows\n"
            output += " -> Testing Set: 400 rows"

        # Step 8: Scaling
        elif step_id == 8:
            # We need X_train to show scaling
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.model_selection import train_test_split

            df = pd.read_csv(DATA_PATH)
            for col in ["Provider", "Server_Type", "Region", "OS_Type"]:
                df[col] = LabelEncoder().fit_transform(df[col])

            X = df.drop(columns=["CPU_Load"])
            y = df["CPU_Load"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            output = ">>> scaler = StandardScaler()\n"
            output += ">>> X_train = scaler.fit_transform(X_train)\n"
            output += ">>> X_test = scaler.transform(X_test)\n\n"
            output += "First 3 Scaled Rows:\n" + str(X_train_scaled[:3])

        # Step 9: Training
        elif step_id == 9:
            output = (
                ">>> model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
            )
            output += ">>> model.fit(X_train, y_train)\n\n"

            # Simulate delay
            time.sleep(1.0)
            output += "✅ Model training complete."

        # Step 10: Evaluation
        elif step_id == 10:
            # We need a REAL model instance here to predict
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_absolute_error, r2_score

            # Quick Re-Build Pipeline
            df = pd.read_csv(DATA_PATH)
            for col in ["Provider", "Server_Type", "Region", "OS_Type"]:
                df[col] = LabelEncoder().fit_transform(df[col])
            X = df.drop(columns=["CPU_Load"])
            y = df["CPU_Load"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Real Eval
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            output = ">>> predictions = model.predict(X_test)\n"
            output += ">>> print(f'MAE: {mean_absolute_error(y_test, predictions)}')\n"
            output += ">>> print(f'R2: {r2_score(y_test, predictions)}')\n\n"
            output += "Model Evaluation:\n"
            output += f" -> Mean Absolute Error: {mae:.2f}\n"
            output += f" -> R2 Score: {r2*100:.2f}%"

        # Step 11: Save
        elif step_id == 11:
            output = ">>> pickle.dump(model, open('model.pkl', 'wb'))\n"
            output += ">>> pickle.dump(scaler, open('scaler.pkl', 'wb'))\n"
            output += ">>> pickle.dump(encoders, open('encoders.pkl', 'wb'))\n\n"
            output += "Files saved: model.pkl, scaler.pkl, encoders.pkl"

        return jsonify({"output": output})

    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})


# --- 5. PREDICT ROUTE ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data"})
        processed, err = process_input(data)
        if err:
            return jsonify({"error": err})

        pred = main_model.predict(processed)[0]
        status = "Normal"
        if pred > 75:
            status = "High Load ⚠️"
        if pred > 90:
            status = "Critical 🚨"

        return jsonify({"cpu_load": round(pred, 2), "status": status})
    except Exception as e:
        return jsonify({"error": str(e)})


# --- 6. DOWNLOAD ROUTES ---
@app.route("/download-model")
def download_model():
    return send_file("model.pkl", as_attachment=True)


@app.route("/download-scaler")
def download_scaler():
    return send_file("scaler.pkl", as_attachment=True)


@app.route("/download-encoders")
def download_encoders():
    return send_file("encoders.pkl", as_attachment=True)


@app.route("/download-data")
def download_data():
    """Download the CSV Dataset"""
    try:
        return send_file(DATA_PATH, as_attachment=True)
    except Exception as e:
        return str(e)


@app.route("/download-script")
def download_script():
    """Download the Python Training Script"""
    try:
        return send_file(SCRIPT_PATH, as_attachment=True)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
