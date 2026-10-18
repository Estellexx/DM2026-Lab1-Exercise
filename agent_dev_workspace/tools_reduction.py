from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for dimensionality reduction and label processing.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: Optional[float] = None,
        n_neighbors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Reduces feature_matrix dimensions using PCA, t-SNE, or UMAP, and attaches a colored scatter plot to session.pending_figure.

        Args:
            method: Method to use for reduction ('pca', 'tsne', or 'umap'). Default: 'pca'.
            n_components: Number of dimensions to reduce to (default: 2).
            random_state: Seed for reproducible output (optional).
            perplexity: t-SNE perplexity parameter (optional, useful for small datasets).
            n_neighbors: UMAP n_neighbors parameter (optional, useful for small datasets).

        Returns:
            A summary dictionary containing reduced coordinates and variance ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No DTM feature matrix found. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray()
        n_samples = X.shape[0]

        explained_variance_ratio = None

        if method == "pca":
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(X)
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        elif method == "tsne":
            p_val = perplexity if perplexity is not None else min(30.0, max(1.0, float(n_samples - 1)))
            tsne = TSNE(
                n_components=n_components,
                perplexity=p_val,
                random_state=random_state,
            )
            coords = tsne.fit_transform(X)
        elif method == "umap":
            nn_val = n_neighbors if n_neighbors is not None else min(15, max(2, n_samples - 1))
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=nn_val,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Invalid reduction method '{method}'."}

        coords_list = coords.tolist()

        # Store reduced coordinates in artifacts
        session.artifacts[f"reduced_{method}"] = coords_list

        # Build colored scatter plot and assign to session.pending_figure
        fig, ax = plt.subplots(figsize=(7, 5))

        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels

        if labels is not None and len(labels) == n_samples:
            plot_df = pd.DataFrame(
                {"Dim 1": coords[:, 0], "Dim 2": coords[:, 1], "Category": labels}
            )
            sns.scatterplot(
                x="Dim 1", y="Dim 2", hue="Category", data=plot_df, ax=ax, s=70
            )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=70)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1" if method == "pca" else "Dim 1")
        ax.set_ylabel("Component 2" if method == "pca" else "Dim 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": n_samples,
            "coordinates": coords_list,
            "status": "success",
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = [
                round(v, 4) for v in explained_variance_ratio
            ]

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
            full_report={"coordinates": coords_list},
        )
        return summary

    @tool
    def binarize_labels_tool(
        category_column: str = "category_name",
    ) -> Dict[str, Any]:
        """
        One-hot encodes/binarizes category labels in session.dataframe and stores matrix in session.artifacts.

        Args:
            category_column: Name of category label column in session.dataframe (default: 'category_name').

        Returns:
            A summary dictionary containing class labels and binarized matrix shape.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if category_column not in df.columns:
            return {"error": f"Column '{category_column}' not found in dataframe."}

        labels_series = df[category_column]
        one_hot_df = pd.get_dummies(labels_series, dtype=int)

        class_names = one_hot_df.columns.tolist()
        matrix = one_hot_df.values.tolist()

        session.artifacts["binarized_labels"] = {
            "classes": class_names,
            "matrix": matrix,
        }

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "category_column": category_column,
            "n_documents": len(df),
            "classes": class_names,
            "n_classes": len(class_names),
            "status": "success",
        }

        session.store_result(
            "binarize_labels_tool",
            {"category_column": category_column},
            summary,
            full_report={"one_hot_matrix": matrix},
        )
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
