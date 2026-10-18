from typing import Any, Dict, Optional
import re
import pandas as pd
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

    return [tokenize_tool]
