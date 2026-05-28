"""
fetch_prepare_data_price.py

Datenvorbereitung für die Preis-Pipeline:
- Zeitspalten-Vereinheitlichung
- Feature Engineering
- Cleaning

Funktionen sind so gehalten, dass sie direkt in Notebooks oder im ETL-Kontext genutzt werden können.
"""
import pandas as pd

def normalize_time_column(in_df: pd.DataFrame, col: str = "time_utc", to_utc: bool = True, freq: str = "min") -> pd.DataFrame:
    """
    Vereinheitlicht eine Zeitspalte:
    - Konvertiert zu pandas.Timestamp
    - Optional in UTC
    - Rundet auf gewünschte Auflösung (default: Minute)

    Args:
        in_df: DataFrame mit Zeitspalte (wird nicht verändert)
        col: Name der Zeitspalte
        to_utc: True → in UTC konvertieren
        freq: 'H' (Stunde), 'min' (Minute), '15min', etc.
    Returns:
        out_df: DataFrame mit vereinheitlichter Zeitspalte
    """
    out_df = in_df.copy()
    out_df[col] = pd.to_datetime(out_df[col], utc=to_utc)
    out_df[col] = out_df[col].dt.floor(freq)
    return out_df
