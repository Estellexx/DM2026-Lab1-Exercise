from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for feature filtering and correlation analysis.
    """

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """
        Calculates per-term variance on session.feature_matrix and flags near-constant terms below threshold for removal.

        Args:
            threshold: Variance cutoff threshold (default: 0.0). Terms with variance below this are filtered out.

        Returns:
            A summary dictionary detailing kept/removed term counts and variance statistics.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        dense_matrix = X.toarray()
        variances = np.var(dense_matrix, axis=0)

        report_df = pd.DataFrame(
            {"term": feature_names, "variance": variances}
        ).sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["variance"] >= threshold]
        removed_df = report_df[report_df["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms_sample": kept_terms[:20],
            "status": "success",
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def pearson_filter_tool(
        target_class: str, threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates Pearson correlation between term counts and a binary target class indicator.

        Args:
            target_class: The category label to correlate term counts against.
            threshold: Correlation magnitude cutoff threshold (default: 0.0).

        Returns:
            A summary dictionary detailing term correlation score counts and filtered terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        if session.labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        if session.labels is None:
            return {"error": "No category labels found in session. Set labels first."}

        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.labels

        y = (labels == target_class).astype(float)

        dense_matrix = X.toarray()
        correlations = []

        for col_idx in range(dense_matrix.shape[1]):
            x_col = dense_matrix[:, col_idx]
            if np.std(x_col) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                r_val, _ = pearsonr(x_col, y)
                if np.isnan(r_val):
                    r_val = 0.0
            correlations.append(float(r_val))

        report_df = pd.DataFrame(
            {"term": feature_names, "pearson_r": correlations}
        ).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["pearson_r"] >= threshold]
        removed_df = report_df[report_df["pearson_r"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms_sample": kept_terms[:20],
            "status": "success",
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(
        target_class: str, threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates Spearman rank correlation between term counts and a binary target class indicator.

        Args:
            target_class: The category label to correlate term counts against.
            threshold: Correlation magnitude cutoff threshold (default: 0.0).

        Returns:
            A summary dictionary detailing term Spearman correlation score counts and filtered terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        if session.labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        if session.labels is None:
            return {"error": "No category labels found in session. Set labels first."}

        X = session.feature_matrix
        feature_names = session.feature_names
        labels = session.labels

        y = (labels == target_class).astype(float)

        dense_matrix = X.toarray()
        correlations = []

        for col_idx in range(dense_matrix.shape[1]):
            x_col = dense_matrix[:, col_idx]
            if np.std(x_col) == 0 or np.std(y) == 0:
                r_val = 0.0
            else:
                r_val, _ = spearmanr(x_col, y)
                if np.isnan(r_val):
                    r_val = 0.0
            correlations.append(float(r_val))

        report_df = pd.DataFrame(
            {"term": feature_names, "spearman_r": correlations}
        ).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        kept_df = report_df[report_df["spearman_r"] >= threshold]
        removed_df = report_df[report_df["spearman_r"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms_sample": kept_terms[:20],
            "status": "success",
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
