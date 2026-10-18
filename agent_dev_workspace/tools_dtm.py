from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for building and analyzing document-term matrices.
    """

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        label_column: str = "label",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) using CountVectorizer and stores feature_matrix in session.

        Args:
            text_column: Name of text column in session.dataframe (default: 'text').
            label_column: Name of target label column in session.dataframe (default: 'label').
            ngram_range: List of 2 ints [min_n, max_n] for n-gram range (default: [1, 1]).
            min_df: Minimum document frequency for vocabulary inclusion (default: 1).
            max_df: Maximum document frequency for vocabulary inclusion (default: 1.0).

        Returns:
            A summary dictionary containing matrix dimensions, vocabulary size, and sparsity percentage.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        texts = df[text_column].fillna("").tolist()

        # Convert list parameter to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range) if isinstance(ngram_range, list) else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
        )

        try:
            X = vectorizer.fit_transform(texts)
        except ValueError as e:
            return {"error": f"CountVectorizer failed: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state according to pipeline contract
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if label_column in df.columns:
            session.set_labels(df[label_column].values)

        # Calculate matrix sparsity
        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements))
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "vocabulary": feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "status": "success",
        }

        session.store_result(
            "build_dtm_tool",
            {
                "text_column": text_column,
                "label_column": label_column,
                "ngram_range": ngram_range,
                "min_df": min_df,
                "max_df": max_df,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents from session.feature_matrix.

        Args:
            top_n: Optional int to limit returned terms to top N highest frequency terms.

        Returns:
            A summary dictionary containing term frequencies and total terms analyzed.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        # Sum column frequencies across all documents
        term_counts = np.asarray(X.sum(axis=0)).ravel()

        term_freq_map = {
            term: int(count) for term, count in zip(feature_names, term_counts)
        }

        # Sort by count descending
        sorted_frequencies = dict(
            sorted(term_freq_map.items(), key=lambda item: item[1], reverse=True)
        )

        if top_n is not None and top_n > 0:
            displayed_frequencies = dict(list(sorted_frequencies.items())[:top_n])
        else:
            displayed_frequencies = sorted_frequencies

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "n_terms": len(feature_names),
            "term_frequencies": displayed_frequencies,
            "raw_frequencies": term_freq_map,
            "status": "success",
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """
        Generates a heatmap of a positional slice of the document-term matrix.

        Args:
            n_terms: Number of terms (columns) to slice from start (default: 20).
            n_documents: Number of documents (rows) to slice from start (default: 20).

        Returns:
            A summary dictionary containing matrix slice counts and labels, attaching plot to session.pending_figure.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        total_docs, total_terms = X.shape
        row_limit = min(n_documents, total_docs)
        col_limit = min(n_terms, total_terms)

        # Slice submatrix
        slice_matrix = X[:row_limit, :col_limit].toarray()
        sliced_terms = feature_names[:col_limit]
        sliced_doc_ids = [f"Doc_{i}" for i in range(row_limit)]

        # Render heatmap
        fig, ax = plt.subplots(
            figsize=(max(6, col_limit * 0.6), max(4, row_limit * 0.4))
        )
        sns.heatmap(
            slice_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sliced_terms,
            yticklabels=sliced_doc_ids,
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Raw Counts Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "sliced_documents": row_limit,
            "sliced_terms": col_limit,
            "terms": sliced_terms,
            "matrix_slice": slice_matrix.tolist(),
            "status": "success",
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
