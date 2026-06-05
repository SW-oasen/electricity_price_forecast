import pandas as pd
from src.fetch_price_data import prepare_data_for_price_prediction_tomorrow
from datetime import datetime, timedelta

def test_prep():
    print("Testing price prep for tomorrow...")
    
    try:
        df = prepare_data_for_price_prediction_tomorrow(history_days=7)
        print("Success!")
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Check for crucial features
        expected_cols = ['price_de_lu_eur_mwh_lag_24h', 'demand_input_mwh', 'residual_load_input_mwh']
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns {missing}")
        else:
            print("Crucial features are present.")
            
        # Verify custom demand prediction was used
        # In build_price_feature_base, demand_input_mwh falls back to smard_forecast_mwh
        # which we injected with our custom prediction.
        print(f"\nDemand input sample (first 3h):")
        print(df[['time', 'demand_input_mwh']].head(3))
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prep()
