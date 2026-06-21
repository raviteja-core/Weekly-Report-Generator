import unittest

import pandas as pd

from services.csv_service import prepare_dataset
from services.data_quality import validate_and_clean_data
from services.kpi_service import calculate_kpis
from services.mcp_service import handle_mcp_request
from services.schema_service import detect_schema


class TrustFirstPipelineTests(unittest.TestCase):
    def test_validate_and_clean_data_removes_duplicates_and_normalizes_categories(self):
        df = pd.DataFrame(
            {
                "Order Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "Revenue": ["100", "100", "bad"],
                "Category": [" electronics ", "Electronics", "furniture"],
                "Region": [" north", "North ", "south"],
            }
        )

        result = validate_and_clean_data(df)
        clean_df = result["clean_df"]
        quality_report = result["quality_report"]

        self.assertEqual(quality_report["duplicate_rows"], 1)
        self.assertAlmostEqual(quality_report["duplicate_percent"], 0.3333, places=4)
        self.assertEqual(quality_report["invalid_numeric"]["revenue"], 1)
        self.assertEqual(clean_df["category"].tolist(), ["Electronics", "Furniture"])
        self.assertEqual(clean_df["region"].tolist(), ["North", "South"])

    def test_detect_schema_requires_high_confidence(self):
        df = pd.read_csv("sample_data/trust_pipeline_sample.csv")
        clean_df = validate_and_clean_data(df)["clean_df"]
        schema = detect_schema(clean_df)

        self.assertEqual(schema["date_col"], "order date")
        self.assertEqual(schema["revenue_col"], "revenue")
        self.assertGreaterEqual(schema["confidence"], 0.8)

    def test_prepare_dataset_fails_when_duplicate_threshold_exceeded(self):
        csv_text = "\n".join(
            [
                "date,revenue,category,region",
                "2024-01-01,100,A,North",
                "2024-01-01,100,A,North",
                "2024-01-02,120,A,North",
                "2024-01-03,130,B,South",
            ]
        )

        result = prepare_dataset(csv_text)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "Data quality too low")

    def test_calculate_kpis_uses_cleaned_dataframe_and_returns_anomaly_dates(self):
        with open("sample_data/trust_pipeline_sample.csv", encoding="utf-8") as handle:
            csv_text = handle.read()
        dataset = prepare_dataset(csv_text)

        self.assertEqual(dataset["status"], "success")
        kpis = calculate_kpis(dataset["dataframe"])

        self.assertEqual(kpis["total_revenue"], 920.0)
        self.assertEqual(kpis["average_revenue"], 184.0)
        self.assertEqual(kpis["row_count"], 5)
        self.assertIn("2024-01-04", kpis["anomalies"]["dates"])

    def test_mcp_tools_contract_supports_initialize_list_and_call(self):
        with open("sample_data/trust_pipeline_sample.csv", encoding="utf-8") as handle:
            csv_text = handle.read()
        dataset = prepare_dataset(csv_text)

        initialize_response = handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(initialize_response["result"]["serverInfo"]["name"], "reportgenie-mcp")

        list_response = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = {tool["name"] for tool in list_response["result"]["tools"]}
        self.assertIn("calculate_kpis", tool_names)
        self.assertIn("generate_chart", tool_names)

        call_response = handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "calculate_kpis", "arguments": {"period": "D"}},
            },
            dataset=dataset,
        )
        self.assertFalse(call_response["result"]["isError"])
        self.assertEqual(call_response["result"]["structuredContent"]["total_revenue"], 920.0)


if __name__ == "__main__":
    unittest.main()
