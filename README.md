#  ML Lab: Server CPU Load Predictor

An educational Machine Learning platform that simulates the end-to-end process of training a CPU Load prediction model.

##  Overview
This is not just a prediction tool; it is an **interactive laboratory**. The application guides users through:
1.  **Data Loading:** Reading server logs.
2.  **Preprocessing:** Encoding categorical data (OS, Region) and scaling features.
3.  **Training:** Building a Random Forest Regressor in real-time.
4.  **Prediction:** Forecasting CPU load based on server metrics.

##  Key Features
- **Simulation Engine:** Step-by-step execution of the ML pipeline via the UI.
- **Real-time Visualization:** Generates Regression and Feature Importance plots dynamically.
- **Download Hub:** Users can download the trained model (`.pkl`), scaler, and dataset.
- **REST API:** Includes a `/predict` endpoint for external integrations.

##  Tech Stack
- **Backend:** Flask (Python)
- **ML Core:** Scikit-Learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

##  Project Structure
```text
├── app.py              # Main Flask Application (Simulation Logic)
├── train_model.py      # Standalone training script
├── templates/          # HTML Frontend
├── static/             # Generated plots and assets
└── server_data_advanced.csv  # Dataset