import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    # Known Answer #7 check
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert res["status"] == "success"
    assert session.pending_figure is not None

    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]

    actual_coords = np.array(res["coordinates"])
    expected_arr = np.array(expected_coords)

    np.testing.assert_allclose(np.abs(actual_coords), np.abs(expected_arr), atol=1e-3)

    expected_var = [0.7532, 0.1965]
    np.testing.assert_allclose(
        res["explained_variance_ratio"], expected_var, atol=1e-3
    )


def test_reduce_dimensions_tsne():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    res = reduce_dimensions_tool.invoke(
        {"method": "tsne", "perplexity": 2, "random_state": 42}
    )

    assert res["status"] == "success"
    assert session.pending_figure is not None

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)

    np.testing.assert_allclose(coords[5], coords[6], atol=1e-3)


def test_reduce_dimensions_umap():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    res = reduce_dimensions_tool.invoke(
        {"method": "umap", "n_neighbors": 3, "random_state": 42}
    )

    assert res["status"] == "success"
    assert session.pending_figure is not None

    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ]

    actual_coords = np.array(res["coordinates"])
    expected_arr = np.array(expected_coords)
    np.testing.assert_allclose(actual_coords, expected_arr, atol=0.5)


def test_binarize_labels_tool():
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

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(
        t for t in reduction_tools if t.name == "binarize_labels_tool"
    )

    res = binarize_labels_tool.invoke({"category_column": "category_name"})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["classes"] == ["catA", "catB"]
    assert res["n_classes"] == 2

    matrix = np.array(res["one_hot_matrix"])
    assert matrix.shape == (8, 2)

    # First 3 rows + last row are catA ([1, 0]), middle rows are catB ([0, 1])
    assert list(matrix[0]) == [1, 0]
    assert list(matrix[3]) == [0, 1]
