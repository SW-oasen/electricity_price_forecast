import unittest

from src.fetch_price_data import prepare_price_model_dataset


class TestFetchPriceData(unittest.TestCase):
    def test_prepare_price_model_dataset_smoke(self):
        df_base, df_price_model, feature_cols = prepare_price_model_dataset()

        self.assertGreater(len(df_base), 0)
        self.assertGreater(len(df_price_model), 0)
        self.assertGreater(len(feature_cols), 0)

        expected_cols = {
            "demand_input_mwh",
            "gen_wind_input_mwh",
            "gen_pv_input_mwh",
            "residual_load_input_mwh",
            "price_de_lu_eur_mwh",
            "time",
        }
        self.assertTrue(expected_cols.issubset(set(df_price_model.columns)))


if __name__ == "__main__":
    unittest.main()
