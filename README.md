 Personality Predictor
A machine learning web application that predicts a person's personality type (Extrovert, Introvert, or Ambivert) based on 29 self‑reported trait scores. The model is a Random Forest classifier trained on a synthetic dataset, and the whole system is served through a clean, responsive Flask interface.

https://via.placeholder.com/800x400.png?text=Personality+Predictor+Screenshot
(Add a real screenshot here if you have one)

✨ Features
Fast & Accurate: A baseline Random Forest achieves 99.6% accuracy on the synthetic dataset – no lengthy hyperparameter tuning required.

Easy Input: 29 input fields (0‑10) for each personality trait.

Instant Prediction: Submit the form and get your result immediately.

Deploy‑Ready: Simple configuration for hosting on Render, PythonAnywhere, or any cloud platform.

Modular Code: Clean separation between model training, prediction logic, and web interface.

📊 Dataset
The synthetic dataset (personality_synthetic_dataset (1).csv) contains 29 numeric features (e.g., social_energy, talkativeness, empathy, etc.) and a target column personality_type with three classes:

Extrovert

Introvert

Ambivert

Each feature is a float between 0 and 10, representing the intensity of that trait.
(The dataset was artificially generated for educational purposes.)

🛠️ Model Training (Baseline)
We intentionally skip heavy hyperparameter tuning because the baseline model already performs excellently. The training script train_model_baseline.py:

Loads the CSV.

Splits data into train/test (80/20).

Scales features using StandardScaler.

Trains a RandomForestClassifier with default parameters.

Evaluates on the test set.

Saves the model, scaler, and label encoder as .pkl files for the web app.
