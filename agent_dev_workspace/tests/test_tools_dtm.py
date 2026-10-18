import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
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

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke(
        {"text_column": "text", "label_column": "label"}
    )

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5

    # Known Answer #4 check
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert res["vocabulary"] == expected_vocab
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Session state verification
    assert session.feature_matrix is not None
    assert session.feature_names == expected_vocab
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
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

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(
        t for t in dtm_tools if t.name == "term_frequency_tool"
    )

    # Build DTM first
    build_dtm_tool.invoke({"text_column": "text", "label_column": "label"})

    # Known Answer #14 check
    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
    assert res["raw_frequencies"] == expected_counts


def test_dtm_heatmap_tool():
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

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({"text_column": "text", "label_column": "label"})

    # Known Answer #21 check: n_terms=2, n_documents=3
    res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res["status"] == "success"
    assert res["sliced_documents"] == 3
    assert res["sliced_terms"] == 2
    assert res["terms"] == ["alpha", "always"]
    assert res["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
