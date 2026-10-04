"""Screener source labels: Nightly / Manual load / Live / Local."""
from ui.services import _origin_from_meta, source_display


def test_source_display_live():
    d = source_display("live", "live")
    assert d["short"] == "Live scan"
    assert "Yahoo" in d["help"] or "live" in d["help"].lower()


def test_source_display_nightly():
    d = source_display("nightly", "precomputed")
    assert d["short"] == "Nightly"
    assert "Nightly" in d["title"] or "github" in d["help"].lower()


def test_source_display_manual():
    d = source_display("manual", "precomputed")
    assert d["short"] == "Manual load"
    assert "Load" in d["help"] or "clicked" in d["help"].lower()


def test_source_display_local():
    d = source_display("local", "precomputed")
    assert d["short"] == "Local job"


def test_origin_from_meta_auto_nightly():
    assert _origin_from_meta({"runner": "github-actions"}, load_mode="auto") == "nightly"


def test_origin_from_meta_manual_overrides_runner():
    # User button always "manual" even if file was built by Actions
    assert _origin_from_meta({"runner": "github-actions"}, load_mode="manual") == "manual"


def test_origin_from_meta_local():
    assert _origin_from_meta({"runner": "local"}, load_mode="auto") == "local"


def test_origin_from_meta_unknown():
    assert _origin_from_meta({}, load_mode="auto") == "precomputed"
