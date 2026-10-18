import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "label_column": "label"})

    # Initialize filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    # Known Answer #5 check: threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["kept_terms_count"] == 4
    assert res["removed_terms_count"] == 1
    assert res["kept_terms"] == ["alpha", "gamma", "delta", "beta"]
    assert res["removed_terms"] == ["always"]

    # Verify stored full_report in session.results_store
    stored_result = session.results_store[res["result_id"]]
    assert "full_report" in stored_result
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "variance"]

    # Check exact variances from Known Answer #5
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }
    for _, row in full_report.iterrows():
        term = row["term"]
        var_val = row["variance"]
        assert var_val == pytest.approx(expected_variances[term], abs=1e-4)


def test_pearson_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "label_column": "label"})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    # Known Answer #6 check for target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["status"] == "success"
    assert res["total_terms"] == 5

    stored_result = session.results_store[res["result_id"]]
    assert "full_report" in stored_result
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "pearson_r"]

    # Known Answer #6 Pearson r expected values
    expected_pearson = {
        "delta": 1.0000,
        "gamma": 0.8307,
        "always": 0.3780,
        "alpha": -0.6882,
        "beta": -0.7746,
    }

    for _, row in full_report.iterrows():
        term = row["term"]
        r_val = row["pearson_r"]
        assert r_val == pytest.approx(expected_pearson[term], abs=1e-3)


def test_spearman_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "label_column": "label"})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Known Answer #6 check for target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["status"] == "success"
    assert res["total_terms"] == 5

    stored_result = session.results_store[res["result_id"]]
    assert "full_report" in stored_result
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns) == ["term", "spearman_r"]

    # Known Answer #6 Spearman r expected values
    expected_spearman = {
        "delta": 1.0000,
        "gamma": 0.9363,
        "always": 0.3780,
        "alpha": -0.7500,
        "beta": -0.7746,
    }

    for _, row in full_report.iterrows():
        term = row["term"]
        r_val = row["spearman_r"]
        assert r_val == pytest.approx(expected_spearman[term], abs=1e-3)
