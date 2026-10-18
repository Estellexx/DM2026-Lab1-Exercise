import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool():
    # Set up shared session state
    session = SessionState()

    # Load premade dataset tool to load the test fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load sample_fixture.csv
    load_result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Initialize tools under test using the same session
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Run tokenize_tool
    res = tokenize_tool.invoke({"text_column": "text"})

    assert res["status"] == "success"
    assert res["n_documents"] == 8

    # Check session dataframe unigrams column against Known Answer 10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens

    # Check token counts
    total_expected_tokens = sum(len(toks) for toks in expected_tokens)
    assert res["total_tokens"] == total_expected_tokens
    assert res["avg_tokens_per_doc"] == pytest.approx(
        total_expected_tokens / 8, abs=1e-3
    )


def test_check_missing_tool():
    # Fresh session state per test function
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

    data_tools = make_tools(session)
    check_missing_tool = next(
        t for t in data_tools if t.name == "check_missing_tool"
    )

    res = check_missing_tool.invoke({"text_column": "text"})

    # Known Answer #2: exactly 1 row has missing/empty text (index 7)
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_pct"] == pytest.approx(12.5, abs=1e-3)
    assert res["missing_indices"] == [7]


def test_list_files_tool():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known Answer #11: list_files_tool(subdirectory="newdataset")
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_duplicates_tool():
    # Fresh session state so earlier mutations don't interfere
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

    data_tools = make_tools(session)
    check_duplicates_tool = next(
        t for t in data_tools if t.name == "check_duplicates_tool"
    )

    # Check without dropping first
    res_check = check_duplicates_tool.invoke({"text_column": "text", "drop": False})

    # Known Answer #3: 1 duplicate copy
    assert res_check["status"] == "success"
    assert res_check["initial_documents"] == 8
    assert res_check["duplicate_count"] == 1
    assert res_check["remaining_documents"] == 8

    # Now run with drop=True
    res_drop = check_duplicates_tool.invoke({"text_column": "text", "drop": True})

    # Known Answer #3: dropping with keep=False leaves exactly 6 rows
    assert res_drop["status"] == "success"
    assert res_drop["remaining_documents"] == 6
    assert len(session.dataframe) == 6


def test_sample_data_tool():
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

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    # Known Answer #13: n=4, random_state=42 returns rows [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["status"] == "success"
    assert res["sample_size"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_describe_data_tool():
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

    data_tools = make_tools(session)
    describe_data_tool = next(
        t for t in data_tools if t.name == "describe_data_tool"
    )

    res = describe_data_tool.invoke(
        {"text_column": "text", "category_column": "category_name"}
    )

    assert res["status"] == "success"

    # Known Answer #18 stats check
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["max"] == 30

    # Per-category check
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)

    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)

    # Figure set on session
    assert session.pending_figure is not None


def test_inspect_data_tool():
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

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n": 3, "head": True})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["peek_count"] == 3
    assert len(res["rows"]) == 3
    assert "text" in res["columns"]
