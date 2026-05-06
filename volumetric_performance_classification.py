"""
Enhanced ModelOp Center monitor: binary classification performance + dimensional volumetrics.

Combines:
  - OOTB performance monitor (classification_metrics via ModelEvaluator)
  - Additional sklearn/pandas performance metrics (MCC, Cohen's Kappa, per-class precision/recall)
  - OOTB volumetric monitor (total record count)
  - Dimensional record-count tables for Document_Type and action columns
    rendered as ModelOp `generic_table` artifacts
"""

import pandas
import numpy as np

from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

import modelop.monitors.performance as performance
import modelop.monitors.volumetrics as volumetrics
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# Dimensional columns to slice record counts on
DIMENSIONAL_COLUMNS = ["Document_Type", "action"]


# modelop.init
def init(job_json: dict) -> None:
    """Receive the job JSON and validate schema fail-fast.

    Args:
        job_json (dict): job JSON
    """
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(dataframe: pandas.DataFrame) -> dict:
    """Compute binary classification metrics, supplemental performance metrics,
    total volumetrics, and dimensional record-count tables.

    Args:
        dataframe (pandas.DataFrame): Sample (prod) dataset containing scores
            (model outputs), labels (ground truths), and dimensional columns.

    Yields:
        dict: Combined performance + volumetric + dimensional results.
    """

    # ------------------------------------------------------------------
    # 1. OOTB Classification Performance  (ModelEvaluator)
    # ------------------------------------------------------------------
    model_evaluator = performance.ModelEvaluator(
        dataframe=dataframe, job_json=JOB
    )
    performance_results = model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )

    # ------------------------------------------------------------------
    # 2. Supplemental Performance Metrics  (pandas + sklearn)
    # ------------------------------------------------------------------
    # Resolve the score and label column names from the schema
    score_col = infer.get_score_column(JOB)
    label_col = infer.get_label_column(JOB)

    # Build a clean subset: drop rows where either column is null
    perf_df = dataframe[[score_col, label_col]].dropna()

    y_true = perf_df[label_col]
    y_pred = perf_df[score_col]

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class precision, recall, f1, support
    labels_sorted = sorted(y_true.unique())
    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_sorted, zero_division=0
    )

    per_class_table = []
    for i, label in enumerate(labels_sorted):
        per_class_table.append(
            {
                "class": str(label),
                "precision": round(float(prec_arr[i]), 4),
                "recall": round(float(rec_arr[i]), 4),
                "f1_score": round(float(f1_arr[i]), 4),
                "support": int(sup_arr[i]),
            }
        )

    supplemental_performance = {
        "matthews_corrcoef": round(float(mcc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "per_class_metrics": per_class_table,  # generic_table artifact
    }

    # ------------------------------------------------------------------
    # 3. OOTB Volumetrics  (total record count)
    # ------------------------------------------------------------------
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=dataframe)
    volumetric_results = volumetric_monitor.count(job_json=JOB)

    # ------------------------------------------------------------------
    # 4. Dimensional Record Counts  (volumetrics + pandas value_counts)
    # ------------------------------------------------------------------
    dimensional_tables = {}

    for col in DIMENSIONAL_COLUMNS:
        if col not in dataframe.columns:
            logger.warning("Dimensional column '%s' not found — skipping.", col)
            continue

        counts = (
            dataframe[col]
            .fillna("(missing)")
            .value_counts()
            .sort_index()
        )

        # Build a generic_table-compatible list of dicts
        table_rows = [
            {col: str(category), "record_count": int(count)}
            for category, count in counts.items()
        ]

        dimensional_tables[f"record_count_by_{col}"] = table_rows

    # ------------------------------------------------------------------
    # 5. Assemble & yield combined output
    # ------------------------------------------------------------------
    combined_output = {}

    # Merge OOTB performance results
    combined_output.update(performance_results)

    # Merge supplemental performance metrics
    combined_output["matthews_corrcoef"] = supplemental_performance["matthews_corrcoef"]
    combined_output["cohen_kappa"] = supplemental_performance["cohen_kappa"]
    combined_output["per_class_metrics"] = supplemental_performance["per_class_metrics"]

    # Merge OOTB volumetric results
    combined_output.update(volumetric_results)

    # Merge dimensional tables (each key is a generic_table artifact)
    combined_output.update(dimensional_tables)

    yield combined_output