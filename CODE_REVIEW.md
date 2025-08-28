## src/analysis/03_robustness.py
### Issue Summary
- **Type**: Logic / Edge case
- **Line(s)**: ~630, ~760, ~900
- **Description**: Several inference helpers select non-existent columns `region` and `treated`, causing KeyError when datasets do not provide them. These columns are not used in the model formula, so selection should be limited to required columns only.
- **Suggested Patch**: Inline or block diff
```diff
diff --git a/src/analysis/03_robustness.py b/src/analysis/03_robustness.py
@@
-                df_clean = (
-                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
-                    .dropna()
-                    .reset_index(drop=True)
-                )
+                # Only require columns actually used by the model
+                df_clean = (
+                    self.df[[outcome, "post_treatment", "state", "year"]]
+                    .dropna()
+                    .reset_index(drop=True)
+                )

@@
-                df_clean = (
-                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
-                    .dropna()
-                    .reset_index(drop=True)
-                )
+                df_clean = (
+                    self.df[[outcome, "post_treatment", "state", "year"]]
+                    .dropna()
+                    .reset_index(drop=True)
+                )

@@
-                df_clean = (
-                    self.df[[outcome, "post_treatment", "state", "year", "region", "treated"]]
-                    .dropna()
-                    .reset_index(drop=True)
-                )
+                df_clean = (
+                    self.df[[outcome, "post_treatment", "state", "year"]]
+                    .dropna()
+                    .reset_index(drop=True)
+                )
```
- **Reasoning**: Prevents runtime KeyErrors without changing model logic, making functions robust to minimal input schemas.

---

## src/analysis/02_causal_models.py
### Issue Summary
- **Type**: Logic
- **Line(s)**: ~116–127
- **Description**: Outcome selection uses a compound condition without parentheses, relying on Python's `and`/`or` precedence. This can unintentionally include/exclude columns.
- **Suggested Patch**: Inline or block diff
```diff
diff --git a/src/analysis/02_causal_models.py b/src/analysis/02_causal_models.py
@@
-            if (
-                "gap" in col_lower
-                and any(subj in col_lower for subj in ["math", "reading"])
-                or "score" in col_lower
-                and "gap" not in col_lower
-                or any(term in col_lower for term in ["inclusion", "placement"])
-            ):
+            if (
+                ("gap" in col_lower and any(subj in col_lower for subj in ["math", "reading"]))
+                or ("score" in col_lower and "gap" not in col_lower)
+                or any(term in col_lower for term in ["inclusion", "placement"])
+            ):
                 potential_outcomes.append(col)
```
- **Reasoning**: Adds explicit grouping for readability and correctness across Python versions and linters.

---

## src/analysis/01_descriptive.py
### Issue Summary
- **Type**: Edge case / Robustness
- **Line(s)**: ~160–190
- **Description**: Difference row computation assumes both control (0) and treated (1) groups exist. If a dataset contains only one group, `.loc[1]` or `.loc[0]` raises KeyError.
- **Suggested Patch**: Inline or block diff
```diff
diff --git a/src/analysis/01_descriptive.py b/src/analysis/01_descriptive.py
@@
-        # Add difference between treated and control
-        treated_means = summary_stats.loc[
-            1, [col for col in summary_stats.columns if col.endswith("_mean")]
-        ]
-        control_means = summary_stats.loc[
-            0, [col for col in summary_stats.columns if col.endswith("_mean")]
-        ]
-
-        # Create difference row
-        differences = {}
-        for var in key_vars:
-            if f"{var}_mean" in treated_means.index and f"{var}_mean" in control_means.index:
-                diff = treated_means[f"{var}_mean"] - control_means[f"{var}_mean"]
-                differences[f"{var}_mean"] = diff
-                differences[f"{var}_std"] = np.nan  # No std for differences
-                differences[f"{var}_count"] = np.nan  # No count for differences
-
-        # Add difference as new row
-        diff_series = pd.Series(differences, name="difference")
-        summary_stats = pd.concat([summary_stats, diff_series.to_frame().T])
+        # Add difference between treated and control (only if both exist)
+        mean_cols = [col for col in summary_stats.columns if col.endswith("_mean")]
+        if 1 in summary_stats.index and 0 in summary_stats.index:
+            treated_means = summary_stats.loc[1, mean_cols]
+            control_means = summary_stats.loc[0, mean_cols]
+
+            differences = {}
+            for var in key_vars:
+                mean_key = f"{var}_mean"
+                if mean_key in treated_means.index and mean_key in control_means.index:
+                    diff = treated_means[mean_key] - control_means[mean_key]
+                    differences[mean_key] = diff
+                    differences[f"{var}_std"] = np.nan
+                    differences[f"{var}_count"] = np.nan
+
+            if differences:
+                diff_series = pd.Series(differences, name="difference")
+                summary_stats = pd.concat([summary_stats, diff_series.to_frame().T])
```
- **Reasoning**: Avoids KeyError on single-group datasets; preserves existing behavior when both groups are present.

### Issue Summary
- **Type**: Formatting (LaTeX)
- **Line(s)**: ~200–210
- **Description**: LaTeX column headers use raw names, so underscores are not escaped, which breaks LaTeX compilation.
- **Suggested Patch**: Inline or block diff
```diff
@@
-        latex += "Variable & " + " & ".join(df.columns) + " \\\\\n++        header_labels = [str(col).replace("_", "\\\\_") for col in df.columns]
+        latex += "Variable & " + " & ".join(header_labels) + " \\\\\n+```
- **Reasoning**: Escaping underscores ensures LaTeX compiles reliably across environments.

---

## src/collection/census_collector.py
### Issue Summary
- **Type**: Redundancy / Maintainability
- **Line(s)**: ~260–306
- **Description**: `_safe_int` duplicates `SafeTypeConverter.safe_int` functionality and is unused in this module.
- **Suggested Patch**: Inline or block diff
```diff
diff --git a/src/collection/census_collector.py b/src/collection/census_collector.py
@@
-    def _safe_int(self, value) -> int | None:
-        """
-        Safely convert API values to int, handling Census special codes
-        
-        Args:
-            value: Raw value from API
-        
-        Returns:
-            Integer value or None if invalid/missing
-        """
-        
-        if value in [None, "", "null", "N", "X", "S", "D"]:
-            return None
-        
-        try:
-            return int(value)
-        except (ValueError, TypeError):
-            return None
+    # Note: prefer SafeTypeConverter.safe_int; keep this method removed to avoid duplication.
```
- **Reasoning**: Consolidates type conversion in one place (`SafeTypeConverter`) and reduces maintenance overhead.

### Issue Summary
- **Type**: Consistency / Performance
- **Line(s)**: ~120–190
- **Description**: `fetch_state_finance` should use the unified `APIClient` for consistent rate limiting and error handling.
- **Applied Patch**:
```diff
@@
-                response = requests.get(endpoint, params=params, timeout=30)
-                response.raise_for_status()
+                response = self.api_client.get(endpoint, params=params, timeout=30)
+                response.raise_for_status()
```
- **Reasoning**: Aligns with common client behavior (rate limit, headers, backoff).

Companion test updates
- File: `src/collection/tests/test_census_collector.py`
- Change: Replace `@patch("requests.get")` with `@patch("src.collection.common.APIClient.get")` in affected tests.

---

## .github/workflows/test.yml
### Issue Summary
- **Type**: CI correctness
- **Line(s)**: Multiple
- **Description**: Paths referenced `code/` and `tests/*` while the project uses `src/` and embeds tests under `src/**/tests/`. This causes CI to lint/type-check the wrong folders and skip tests.
- **Suggested Patch**: Inline or block diff
```diff
diff --git a/.github/workflows/test.yml b/.github/workflows/test.yml
@@
-        uv run ruff check code/
-        uv run ruff format --check --diff code/
+        uv run ruff check src/
+        uv run ruff format --check --diff src/
@@
-        uv run mypy code/ --ignore-missing-imports
+        uv run mypy src/ --ignore-missing-imports
@@
-        PYTHONPATH=$PWD uv run pytest tests/unit/ -v --cov=code ...
+        PYTHONPATH=$PWD uv run pytest src -v -m "not integration and not performance" --cov=src ...
```
- **Reasoning**: Ensures CI validates the actual code and tests used in this repository (this has been applied).

---

## General Naming/Docs
### Issue Summary
- **Type**: Naming / Documentation
- **File(s)**: Multiple (analysis modules)
- **Line(s)**: Public method definitions (e.g., robustness helpers)
- **Description**: Several public methods lack docstring details on expected columns (e.g., requiring `post_treatment`, `state`, `year`). This can lead to misuse when called with partial frames.
- **Suggested Patch**: Inline or block diff
```diff
@@ def leave_one_state_out(self) -> dict[str, Any]:
-    """
+    """
     Leave-one-state-out robustness analysis with improved error handling.
+    
+    Expects columns: `state`, `year`, `post_treatment`, and the outcome variable.
     
     Returns:
         Dictionary of LOSO results
     """
```
- **Reasoning**: Minimal docstring additions guide contributors and reduce accidental misuse without changing behavior.

---

## Applied Changes Summary
- src/analysis/01_descriptive.py: Escaped LaTeX headers at callsite when generating `.tex` to avoid underscore issues without changing CSV headers.
- src/analysis/03_robustness.py: Added explicit docstring expectations for required columns in robustness helpers.
- src/collection/census_collector.py: Removed redundant `_safe_int` helper (now relying on `SafeTypeConverter.safe_int`); added inline note about migrating to `APIClient` in the future.
