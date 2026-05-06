"""
Enhanced ModelOp Center monitor: binary classification performance + dimensional volumetrics.

Combines:
  - OOTB performance monitor (classification_metrics via ModelEvaluator)
  - Additional sklearn/pandas performance metrics (MCC, Cohen's Kappa, per-class precision/recall)
  - OOTB volumetric monitor (total record count)
  - Dimensional record-count tables for Document_Type and action columns
    rendered as ModelOp generic_table artifacts
"""

import json
import pandas

from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

import modelop.monitors.performance as performance
import modelop.monitors.volumetrics as volumetrics
import modelop.schema.infer as infer
import modelop.utils as utils

# modelop.init
def init(init_param):
    global JOB
    JOB = json.loads(init_param["rawJson"])
    logger = utils.configure_logger()
    global LABEL_COLUMN
    LABEL_COLUMN=JOB.get("jobParameters", {}).get("LABEL_COLUMN", []) 
    global SCORE_COLUMN
    SCORE_COLUMN=JOB.get("jobParameters", {}).get("SCORE_COLUMN", []) 
    # Dimensional columns to slice record counts on
    global DIMENSIONAL_COLUMNS
    DIMENSIONAL_COLUMNS = ["Document_Type", "action"]

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

    global LABEL_COLUMN
    global SCORE_COLUMN
    print("Running the metrics function") 
    y_label=dataframe[LABEL_COLUMN]
    y_pred=dataframe[SCORE_COLUMN]

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
    perf_df = dataframe[[SCORE_COLUMN, LABEL_COLUMN]].dropna()

    y_true = perf_df[LABEL_COLUMN]
    y_pred = perf_df[SCORE_COLUMN]

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

    # ------------------------------------------------------------------
    # 3. OOTB Volumetrics  (total record count)
    # ------------------------------------------------------------------
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=dataframe)
    volumetric_results = volumetric_monitor.count(job_json=JOB)

    # ------------------------------------------------------------------
    # 4. Dimensional Record Counts  (pandas value_counts)
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

        table_rows = [
            {col: str(category), "record_count": int(count)}
            for category, count in counts.items()
        ]

        dimensional_tables[f"record_count_by_{col}"] = table_rows

    # ------------------------------------------------------------------
    # 5. Assemble & yield combined output
    # ------------------------------------------------------------------
    combined_output = {}

    combined_output.update(performance_results)

    combined_output["matthews_corrcoef"] = round(float(mcc), 4)
    combined_output["cohen_kappa"] = round(float(kappa), 4)
    combined_output["per_class_metrics"] = per_class_table

    combined_output.update(volumetric_results)

    combined_output.update(dimensional_tables)

    yield combined_output