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
