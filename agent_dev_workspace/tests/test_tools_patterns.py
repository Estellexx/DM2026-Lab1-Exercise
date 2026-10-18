import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_fpgrowth_and_zero_terms():
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

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(
        t for t in patterns_tools if t.name == "mine_patterns_tool"
    )

    # Known Answer #17: variance filtering keeps 0 terms cleanly
    res_var = mine_patterns_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "variance",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_var["status"] == "success"
    assert res_var["pattern_count"] == 0
    assert res_var["patterns"] == []

    # Known Answer #17: tfidf filtering with fpgrowth min_sup=1
    res_tfidf_a = mine_patterns_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_tfidf_a["status"] == "success"
    assert res_tfidf_a["pattern_count"] == 1
    assert res_tfidf_a["patterns"][0]["pattern"] == "alpha"
    assert res_tfidf_a["patterns"][0]["support"] == 3

    res_tfidf_b = mine_patterns_tool.invoke(
        {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_tfidf_b["status"] == "success"
    assert res_tfidf_b["pattern_count"] == 1
    assert res_tfidf_b["patterns"][0]["pattern"] == "gamma"
    assert res_tfidf_b["patterns"][0]["support"] == 4


def test_mine_patterns_tool_topk_and_maxfpgrowth():
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

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(
        t for t in patterns_tools if t.name == "mine_patterns_tool"
    )

    # Known Answer #20: topk with k=1
    res_topk_a = mine_patterns_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": 1,
        }
    )
    assert res_topk_a["status"] == "success"
    assert res_topk_a["pattern_count"] == 1
    assert res_topk_a["patterns"][0]["pattern"] == "alpha"
    assert res_topk_a["patterns"][0]["support"] == 3

    # Known Answer #20: maxfpgrowth returns 0 patterns cleanly on single-term data
    res_max_a = mine_patterns_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1,
        }
    )
    assert res_max_a["status"] == "success"
    assert res_max_a["pattern_count"] == 0
    assert res_max_a["patterns"] == []
