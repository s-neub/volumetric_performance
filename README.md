# Enhanced Performance & Volumetrics Monitor: Binary Classification with Dimensional Record Counts

This custom ModelOp Center monitor combines the **OOTB Performance Monitor (Classification)** and the **OOTB Volumetrics Monitor (Count)** into a single unified monitor, and extends them with supplemental performance metrics computed via `pandas` and `scikit-learn`, plus dimensional record-count tables rendered as `generic_table` artifacts in the ModelOp UI.

## Input Assets

| Type | Number | Description |
| --- | --- | --- |
| Baseline Data | **0** |  |
| Comparator Data | **1** | A dataset corresponding to a slice of production data containing model outputs, ground truths, and dimensional columns |

## Assumptions & Requirements

- Underlying `BUSINESS_MODEL` being monitored has a **job JSON** asset.
- `BUSINESS_MODEL` is a **binary classification** model.
- Input data must contain:

    - 1 column with **role=label** (ground truth) and **dataClass=categorical**
    - 1 column with **role=score** (model output) and **dataClass=categorical**
- Input data should contain one or more **dimensional columns** for record-count breakdowns. By default the monitor slices on:

    - `Document_Type` — the type of document being evaluated (e.g., SAF/POCF, Dosage, RFI, CDP Batch Details, Certificate of Analysis, Environmental Monitoring, etc.)
    - `action` — the action taken during evaluation (e.g., `incorrect-evaluation`, `add-evaluation`, `update-message`, `other`)

### Data Schema

| Column | Type | Role |
| --- | --- | --- |
| `batch_id` | string | Identifier |
| `context` | JSON string | Metadata (batch\_id, document\_type, material\_number, arm, document\_section, validation\_rule, page\_number) |
| `id` | integer | Row identifier |
| `action` | categorical | **Dimensional** — action taken during evaluation |
| `feedback` | string | Free-text feedback |
| `feedback_type` | categorical | Feedback classification (e.g., `ai-correction`) |
| `created_at` | timestamp | Record creation time |
| `updated_at` | timestamp | Record update time |
| `user_id` | integer | Evaluator identifier |
| `Document_Type` | categorical | **Dimensional** — document type being evaluated |
| `Model_Output` | categorical (TRUE/FALSE) | **Score** column — model's predicted label |
| `Ground_Truth` | categorical (TRUE/FALSE) | **Label** column — actual label |

## Dependencies

    `modeloppandasnumpyscikit-learn`

## Execution

### 1. Initialization (`init`)

- Accepts the job JSON asset and validates the input schema corresponding to the `BUSINESS_MODEL` being monitored (via `modelop.schema.infer.validate_schema`).

### 2. OOTB Classification Performance

- Instantiates the `performance.ModelEvaluator` class using the input dataframe and job JSON.
- Runs the `classification_metrics` pre-defined test to compute **accuracy**, **AUC**, **precision**, **recall**, **F1 score**, and **confusion matrix**.
- Results are returned in the standard `performance` test list format.

### 3. Supplemental Performance Metrics (pandas + scikit-learn)

- Resolves the score and label columns from the job JSON schema using `modelop.schema.infer`.
- Drops rows where either the score or label column is null (via `pandas.DataFrame.dropna`).
- Computes additional metrics **not included** in the OOTB performance output:

    - **Matthews Correlation Coefficient (MCC)** — `sklearn.metrics.matthews_corrcoef` — a balanced measure that accounts for true/false positives and negatives, useful even with imbalanced classes.
    - **Cohen's Kappa** — `sklearn.metrics.cohen_kappa_score` — measures inter-rater agreement between the model's predictions and ground truth, adjusting for chance agreement.
    - **Per-class Precision, Recall, F1, and Support** — `sklearn.metrics.precision_recall_fscore_support` — provides a per-class breakdown, output as a `generic_table` artifact.

### 4. OOTB Volumetrics (Total Record Count)

- Instantiates the `volumetrics.VolumetricMonitor` class using the input dataframe.
- Runs the `count` test to return the total `record_count`.
- Results are returned in the standard `volumetrics` test list format.

### 5. Dimensional Record Counts (pandas + Volumetrics)

- For each configured dimensional column (`Document_Type`, `action`):

    - Uses `pandas.Series.value_counts()` to compute record counts per category.
    - Fills null values with `"(missing)"` to ensure complete accounting.
    - Outputs results as a list of dicts conforming to the ModelOp [Build a Custom Test or Monitor | Custom Monitor Output Charts, Graphs, Tables](https://modelopdocs.atlassian.net/wiki/spaces/dv33/pages/3159988901/Build+a+Custom+Test+or+Monitor#Custom-Monitor-Output---Charts%2C-Graphs%2C-Tables)`generic_table` format so the UI renders them as tables automatically.

### 6. Combined Output

- All results from steps 2–5 are merged into a single dictionary and yielded.

## Monitor Output

    `{ "accuracy": <accuracy>, "auc": <auc>, "f1_score": <f1_score>, "precision": <precision>, "recall": <recall>, "confusion_matrix": <confusion_matrix>, "performance": [ { "test_category": "performance", "test_name": "Classification Metrics", "test_type": "classification_metrics", "test_id": "performance_classification_metrics", "values": { "accuracy": <accuracy>, "auc": <auc>, "f1_score": <f1_score>, "precision": <precision>, "recall": <recall>, "confusion_matrix": <confusion_matrix> } } ], "matthews_corrcoef": <mcc_value>, "cohen_kappa": <kappa_value>, "per_class_metrics": [ {"class": "FALSE", "precision": <p>, "recall": <r>, "f1_score": <f1>, "support": <n>}, {"class": "TRUE", "precision": <p>, "recall": <r>, "f1_score": <f1>, "support": <n>} ], "record_count": <total_record_count>, "volumetrics": [ { "test_name": "Count", "test_category": "volumetrics", "test_type": "count", "test_id": "volumetrics_count", "values": { "record_count": <total_record_count> } } ], "record_count_by_Document_Type": [ {"Document_Type": "Batch Genealogy", "record_count": <n>}, {"Document_Type": "CDP Batch Details", "record_count": <n>}, {"Document_Type": "Certificate of Analysis", "record_count": <n>}, {"Document_Type": "Deviation Assessment", "record_count": <n>}, {"Document_Type": "Dosage", "record_count": <n>}, {"Document_Type": "Dosage Assignment", "record_count": <n>}, {"Document_Type": "Environmental Monitoring", "record_count": <n>}, {"Document_Type": "PDP Batch Details", "record_count": <n>}, {"Document_Type": "QP Batch Details", "record_count": <n>}, {"Document_Type": "QP/RP Usage Decision", "record_count": <n>}, {"Document_Type": "RFI", "record_count": <n>}, {"Document_Type": "RFI Status", "record_count": <n>}, {"Document_Type": "SAF/POCF", "record_count": <n>}, {"Document_Type": "SAF/POCF Status", "record_count": <n>} ], "record_count_by_action": [ {"action": "add-evaluation", "record_count": <n>}, {"action": "incorrect-evaluation", "record_count": <n>}, {"action": "other", "record_count": <n>}, {"action": "update-message", "record_count": <n>} ]}`

## Output Artifact Summary

| Output Key | Source | UI Rendering |
| --- | --- | --- |
| `accuracy`, `auc`, `f1_score`, `precision`, `recall`, `confusion_matrix` | OOTB `ModelEvaluator` | Standard classification visuals |
| `performance` | OOTB `ModelEvaluator` | Test result list |
| `matthews_corrcoef` | `sklearn.metrics.matthews_corrcoef` | Scalar metric |
| `cohen_kappa` | `sklearn.metrics.cohen_kappa_score` | Scalar metric |
| `per_class_metrics` | `sklearn.metrics.precision_recall_fscore_support` | `generic_table` |
| `record_count` | OOTB `VolumetricMonitor` | Scalar metric |
| `volumetrics` | OOTB `VolumetricMonitor` | Test result list |
| `record_count_by_Document_Type` | `pandas.Series.value_counts` | `generic_table` |
| `record_count_by_action` | `pandas.Series.value_counts` | `generic_table` |

## Customization

### Adding Dimensional Columns

To slice record counts on additional columns, edit the `DIMENSIONAL_COLUMNS` list at the top of the monitor source:
    `DIMENSIONAL_COLUMNS = ["Document_Type", "action", "feedback_type"]`

Any column present in the input dataframe and listed in `DIMENSIONAL_COLUMNS` will produce a corresponding `record_count_by_<column>` table in the output. Missing columns are logged as warnings and skipped.

### Adding Supplemental Metrics

Additional `sklearn` or `pandas`-based metrics can be appended in the "Supplemental Performance Metrics" section of the `metrics` function. Ensure new keys are added to the `combined_output` dictionary so they appear in the yielded result.