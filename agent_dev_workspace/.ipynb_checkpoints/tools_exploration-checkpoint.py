from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for document and feature correlation exploration.
    """

    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> Dict[str, Any]:
        """
        Computes cosine similarity between two documents in session.feature_matrix using raw term counts.

        Args:
            doc_index1: Row index of first document in feature_matrix.
            doc_index2: Row index of second document in feature_matrix.

        Returns:
            A summary dictionary containing calculated cosine similarity score and result_id.
        """
        if session.feature_matrix is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_docs = X.shape[0]

        if doc_index1 < 0 or doc_index1 >= n_docs:
            return {"error": f"doc_index1 {doc_index1} out of bounds (total docs: {n_docs})."}
        if doc_index2 < 0 or doc_index2 >= n_docs:
            return {"error": f"doc_index2 {doc_index2} out of bounds (total docs: {n_docs})."}

        v1 = X[doc_index1].toarray().ravel()
        v2 = X[doc_index2].toarray().ravel()

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            sim_score = 0.0
        else:
            sim_score = float(np.dot(v1, v2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": round(sim_score, 4),
            "status": "success",
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index1": doc_index1, "doc_index2": doc_index2},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """
        Computes Pearson correlation matrix over top N variance terms in session.feature_matrix and attaches a heatmap to session.pending_figure.

        Args:
            top_n: Number of top terms by variance to include in correlation matrix (default: 20).

        Returns:
            A summary dictionary containing top terms, correlation matrix, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray()
        feature_names = session.feature_names

        # Compute per-term variance
        variances = np.var(X, axis=0)

        # Sort terms by variance descending
        sorted_indices = np.argsort(-variances)
        limit = min(top_n, len(feature_names))
        top_indices = sorted_indices[:limit]

        top_terms = [feature_names[i] for i in top_indices]
        top_submatrix = X[:, top_indices]

        # Compute Pearson correlation matrix across terms (columns)
        # np.corrcoef expects variables as rows, so transpose top_submatrix
        corr_matrix = np.corrcoef(top_submatrix.T)

        # Handle NaN values if any term is constant
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Build correlation heatmap and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, limit * 0.8), max(5, limit * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f" if limit > 10 else ".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {limit} Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        # Format correlation matrix for JSON response
        rounded_matrix = [[round(float(val), 4) for val in row] for row in corr_matrix]

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "top_n": limit,
            "terms": top_terms,
            "correlation_matrix": rounded_matrix,
            "status": "success",
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
