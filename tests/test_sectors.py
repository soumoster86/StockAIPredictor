"""Sector heuristics + enrich helper."""
from ui.sectors import classify_sector, enrich_with_sector
import pandas as pd


def test_classify_known_sectors():
    assert classify_sector("HDFC Bank Limited") == "Banking"
    assert classify_sector("Infosys Limited") == "IT"
    assert classify_sector("Sun Pharmaceutical Industries Limited") == "Pharma"
    assert classify_sector("Tata Steel Limited") == "Metal"
    assert classify_sector("Some Random Co") == "Other"
    assert classify_sector("") == "Other"
    assert classify_sector(None) == "Other"


def test_enrich_with_sector_adds_column():
    df = pd.DataFrame({
        "Name": ["HDFC Bank Limited", "Wipro Limited", "Mystery Corp"],
        "Symbol": ["HDFCBANK.NS", "WIPRO.NS", "MYST.NS"],
        "Buy Score": [70, 60, 50],
    })
    out = enrich_with_sector(df)
    assert "Sector" in out.columns
    assert out.loc[0, "Sector"] == "Banking"
    assert out.loc[1, "Sector"] == "IT"
    assert out.loc[2, "Sector"] == "Other"
