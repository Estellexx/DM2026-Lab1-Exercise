import os
import tempfile
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session: Any):
    """
    Factory function returning tools for pattern mining using PAMI.
    """

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent itemsets/patterns for a specific document category using PAMI.

        Args:
            category_name: The category/class name in session.dataframe to mine.
            filtering_method: Method to filter terms ('variance', 'tfidf', or 'term_frequency').
            algorithm: Pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default: 'fpgrowth'.
            min_sup: Minimum support count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to return (required for 'topk').

        Returns:
            A summary dictionary containing mined patterns and support counts.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Load a dataset first."}

        df = session.dataframe
        if "category_name" not in df.columns:
            return {"error": "Column 'category_name' not found in dataframe."}

        # Step 1: Filter to category
        cat_df = df[df["category_name"] == category_name]
        if len(cat_df) == 0:
            return {"error": f"No documents found for category '{category_name}'."}

        texts = cat_df["text"].fillna("").tolist()

        # Build fresh per-category DTM with English stopwords removed
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived stopword removal",
                "status": "success",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived stopword removal",
                "status": "success",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dense_X = X.toarray()

        # Step 2: Apply vocabulary filtering
        if filtering_method == "variance":
            scores = np.var(dense_X, axis=0)
            low_pct = np.percentile(scores, 5)
            high_pct = np.percentile(scores, 95)
            mask = (scores > low_pct) & (scores < high_pct)
        elif filtering_method == "term_frequency":
            scores = np.sum(dense_X, axis=0)
            low_pct = np.percentile(scores, 5)
            high_pct = np.percentile(scores, 95)
            mask = (scores > low_pct) & (scores < high_pct)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(X).toarray()
            scores = np.mean(tfidf_matrix, axis=0)
            pct20 = np.percentile(scores, 20)
            mask = scores >= pct20
        else:
            return {"error": f"Invalid filtering_method '{filtering_method}'."}

        surviving_indices = np.where(mask)[0]
        surviving_terms = [feature_names[i] for i in surviving_indices]

        # Step 3: Handle 0 surviving terms cleanly
        if len(surviving_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived filtering",
                "status": "success",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert surviving terms to transactional DataFrame
        binary_matrix = (dense_X[:, surviving_indices] > 0).astype(int)
        tx_df = pd.DataFrame(binary_matrix, columns=surviving_terms)

        # Remove empty rows (rows with no items)
        tx_df = tx_df[tx_df.sum(axis=1) > 0]

        if len(tx_df) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no non-empty transactions",
                "status": "success",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Convert to PAMI Transactional Database via tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "tx_data.txt")
            out_file = os.path.join(tmpdir, "patterns.txt")

            from PAMI.extras.convert.DF2DB import DF2DB
            converter = DF2DB(tx_df)
            converter.convert2TransactionalDatabase(db_file)

            # Step 5: Execute PAMI mining algorithm
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter required for fpgrowth algorithm."}
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(db_file, minSup=min_sup)
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k parameter required for topk algorithm."}
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(db_file, k=k)
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter required for maxfpgrowth algorithm."}
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
            else:
                return {"error": f"Invalid algorithm '{algorithm}'."}

            miner.startMine()
            miner.save(out_file)

            # Parse pattern output file
            patterns = []
            if os.path.exists(out_file):
                with open(out_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            items_str = parts[0].strip()
                            sup_str = parts[1].strip()
                            try:
                                sup = float(sup_str) if "." in sup_str else int(sup_str)
                            except ValueError:
                                sup = sup_str
                            patterns.append({"pattern": items_str, "support": sup})

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": len(patterns),
            "patterns": patterns,
            "status": "success",
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
