from typing import Any, Dict, Optional
import os
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for data inspection, cleaning, and tokenization.
    """

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """
        Tokenizes document text into lowercased unigram tokens (words) and stores them in session.dataframe.

        Args:
            text_column: Name of the dataframe column containing text to tokenize (default: 'text').

        Returns:
            A summary dictionary containing tokenization statistics and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Simple unigram tokenization: lowercase and find word tokens via regex
        def _tokenize(text: Any) -> list:
            if not isinstance(text, str) or not text.strip():
                return []
            return re.findall(r"\b\w+\b", text.lower())

        df["unigrams"] = df[text_column].apply(_tokenize)

        # Compute summary statistics
        n_documents = len(df)
        doc_token_counts = df["unigrams"].apply(len)
        total_tokens = int(doc_token_counts.sum())
        avg_tokens_per_doc = float(doc_token_counts.mean()) if n_documents > 0 else 0.0

        # Store artifacts
        session.artifacts["unigrams"] = df["unigrams"].tolist()

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens_per_doc, 4),
            "status": "success",
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """
        Checks for missing, null, or whitespace-only text values in session.dataframe.

        Args:
            text_column: Name of the column to check for missing/empty text (default: 'text').

        Returns:
            A summary dictionary containing missing count, missing percentage, and indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Identify missing or empty/whitespace-only values
        is_null = df[text_column].isna()
        is_empty = df[text_column].apply(
            lambda val: isinstance(val, str) and len(val.strip()) == 0
        )
        missing_mask = is_null | is_empty

        missing_indices = df.index[missing_mask].tolist()
        missing_count = len(missing_indices)
        n_documents = len(df)
        missing_pct = float(100.0 * missing_count / n_documents) if n_documents > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 4),
            "missing_indices": missing_indices,
            "status": "success",
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and subdirectories within a repository folder so dataset paths can be discovered.

        Args:
            subdirectory: Relative path of the folder to list (default: root '').

        Returns:
            A summary dictionary with listed 'directories' and 'files'.
        """
        target_dir = Path(subdirectory) if subdirectory else Path(".")

        if not target_dir.exists():
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not target_dir.is_dir():
            return {"error": f"Path '{subdirectory}' is not a directory."}

        directories = []
        files = []

        for item in sorted(target_dir.iterdir()):
            # Skip hidden files/folders
            if item.name.startswith("."):
                continue

            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "status": "success",
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_duplicates_tool(
        text_column: str = "text", drop: bool = False
    ) -> Dict[str, Any]:
        """
        Checks for duplicate documents based on text_column, and optionally drops all copies (keep=False).

        Args:
            text_column: Name of the column to check for duplicates (default: 'text').
            drop: If True, drops all copies of duplicated rows using keep=False (default: False).

        Returns:
            A summary dictionary detailing duplicate count, initial/remaining documents, and action taken.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Count duplicate entries (using keep=False to find every row that is involved in a duplicate)
        dup_mask = df.duplicated(subset=[text_column], keep=False)
        duplicated_rows_count = int(dup_mask.sum())

        # Standard pandas duplicate count (count of extra copies beyond the first)
        num_duplicate_extra_copies = int(
            df.duplicated(subset=[text_column], keep="first").sum()
        )

        initial_documents = len(df)

        if drop and duplicated_rows_count > 0:
            # Match contract: keep=False drops all copies of duplicated rows
            session.dataframe = df.drop_duplicates(
                subset=[text_column], keep=False
            ).reset_index(drop=True)
            remaining_documents = len(session.dataframe)
        else:
            remaining_documents = initial_documents

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "initial_documents": initial_documents,
            "duplicate_count": num_duplicate_extra_copies,
            "duplicated_rows_count": duplicated_rows_count,
            "dropped": drop,
            "remaining_documents": remaining_documents,
            "status": "success",
        }

        session.store_result(
            "check_duplicates_tool",
            {"text_column": text_column, "drop": drop},
            summary,
        )
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Subsamples rows from session.dataframe using absolute count n or fraction frac.

        Args:
            n: Number of items/rows to sample (optional).
            frac: Fraction of items/rows to sample (optional).
            random_state: Seed for reproducible random sampling (optional).

        Returns:
            A summary dictionary containing sampled indices, sample size, and sample rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe

        if n is None and frac is None:
            n = 5  # Default to 5 samples if neither n nor frac specified

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sample_size": len(sampled_df),
            "sampled_indices": sampled_indices,
            "sample_records": sampled_df.to_dict(orient="records"),
            "status": "success",
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text", category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """
        Computes text length describe stats overall and by category, and attaches a box plot to session.pending_figure.

        Args:
            text_column: Name of text column (default: 'text').
            category_column: Name of category label column (default: 'category_name').

        Returns:
            A summary dictionary containing text length stats overall and per category.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if text_column not in df.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Calculate text character lengths matching Master's df["text"].apply(len)
        df["text_length"] = df[text_column].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )

        overall_stats = df["text_length"].describe().to_dict()

        cat_stats = {}
        if category_column in df.columns:
            for cat, group in df.groupby(category_column):
                cat_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Build box plot and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in df.columns:
            sns.boxplot(x=category_column, y="text_length", data=df, ax=ax)
            ax.set_title(f"Document Text Length by {category_column}")
        else:
            sns.boxplot(y="text_length", data=df, ax=ax)
            ax.set_title("Document Text Length Distribution")

        ax.set_ylabel("Text Length (character count)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": cat_stats,
            "status": "success",
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary,
        )
        return summary

    @tool
    def inspect_data_tool(n: int = 5, head: bool = True) -> Dict[str, Any]:
        """
        Peeks at the top or bottom N rows of session.dataframe along with column metadata.

        Args:
            n: Number of rows to peek at (default: 5).
            head: If True peeks at top N rows, if False peeks at bottom N rows (default: True).

        Returns:
            A summary dictionary with dataset size, column names, and row previews.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        peek_df = df.head(n) if head else df.tail(n)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "columns": df.columns.tolist(),
            "peek_mode": "head" if head else "tail",
            "peek_count": len(peek_df),
            "rows": peek_df.to_dict(orient="records"),
            "status": "success",
        }

        session.store_result("inspect_data_tool", {"n": n, "head": head}, summary)
        return summary

    return [
        tokenize_tool,
        check_missing_tool,
        list_files_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
