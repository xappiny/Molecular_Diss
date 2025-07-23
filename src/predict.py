from sklearn.preprocessing import StandardScaler
import pandas as pd
from tabpfn import TabPFNRegressor
import joblib
import numpy as np
from pathlib import Path

DATA_PATH = r'\descriptor_molecular_v2.csv'
SCALER_PATH = r'\scaler.pkl'
TABPFN_MODEL_PATH = r'\tabpfn.pkl'
RF_MODEL_PATH = r'\rf.pkl'
LGBM_MODEL_PATH = r'\lgb.pkl'
OUTPUT_PATH = r'\external_prediction_molecular.csv'

def load_data_and_models():
    try:
        scaler = joblib.load(SCALER_PATH)
        tabpfn_model = joblib.load(TABPFN_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        lgbm_model = joblib.load(LGBM_MODEL_PATH)
        data_new = pd.read_csv(DATA_PATH, encoding='unicode_escape')
        return scaler, tabpfn_model, rf_model, lgbm_model, data_new
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading data or models: {e}")
        raise


def preprocess_data(data_new, scaler):
    try:
        if 'formulation_id' not in data_new.columns:
            raise ValueError("Missing 'formulation_id' column in dataset")

        X_new = data_new.iloc[:, 1:-1]
        if X_new.empty:
            raise ValueError("No features found in dataset (check column selection)")

        X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
        return X_new_scaled
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise


def make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model):
    try:
        predictions = {
            'TabPFN': tabpfn_model.predict(X_new_scaled),
            'RF': rf_model.predict(X_new_scaled),
            'LGBM': lgbm_model.predict(X_new_scaled)
        }
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


def main():
    scaler, tabpfn_model, rf_model, lgbm_model, data_new = load_data_and_models()
    X_new_scaled = preprocess_data(data_new, scaler)
    predictions = make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model)

    data_new['Predicted_diss_TabPFN'] = predictions['TabPFN']
    data_new['Predicted_diss_RF'] = predictions['RF']
    data_new['Predicted_diss_LGBM'] = predictions['LGBM']

    has_output_time = 'Output_time' in data_new.columns
    ASD_predictions = {}

    for formulation_id in data_new['formulation_id'].unique():
        formulation_data = data_new[data_new['formulation_id'] == formulation_id]

        if has_output_time:
            time_points = formulation_data['Output_time'].values
            ASD_predictions[formulation_id] = {
                'TabPFN': dict(zip(time_points, formulation_data['Predicted_diss_TabPFN'].values)),
                'RF': dict(zip(time_points, formulation_data['Predicted_diss_RF'].values)),
                'LGBM': dict(zip(time_points, formulation_data['Predicted_diss_LGBM'].values))
            }
        else:
            ASD_predictions[formulation_id] = {
                'TabPFN': formulation_data['Predicted_diss_TabPFN'].values,
                'RF': formulation_data['Predicted_diss_RF'].values,
                'LGBM': formulation_data['Predicted_diss_LGBM'].values
            }

    print("=== Predicted Dissolution Profiles for ASD Formulations ===")
    for formulation_id, predictions_dict in ASD_predictions.items():
        print(f"Formulation ID: {formulation_id}")
        for model_name, model_predictions in predictions_dict.items():
            print(f"  Model: {model_name}")
            if has_output_time:
                for time_point, diss in model_predictions.items():
                    print(f"    Time: {time_point:.1f} min, Predicted Dissolution: {diss:.3f}%")
            else:
                for idx, diss in enumerate(model_predictions):
                    print(f"    Sample {idx}: Predicted Dissolution: {diss:.3f}%")
        print("-" * 50)

    try:
        predictions_df = pd.DataFrame({
            'formulation_id': data_new['formulation_id'],
            **({'time_point': data_new['Output_time']} if has_output_time else {}),
            'Predicted_diss_TabPFN': predictions['TabPFN'],
            'Predicted_diss_RF': predictions['RF'],
            'Predicted_diss_LGBM': predictions['LGBM']
        })
        predictions_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Predictions saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise


if __name__ == "__main__":
    main()
