"""OOTB ModelOp Center model to compute binary classification metrics and return record count"""
import pandas

import modelop.monitors.performance as performance
import modelop.monitors.volumetrics as volumetrics
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

JOB = {}

# modelop.init
def init(job_json: dict) -> None:
    """A function to receive the job JSON and validate schema fail-fast.

    Args:
        job_json (dict): job JSON
    """

    # Extract job_json and validate
    global JOB
    JOB = job_json
    infer.validate_schema(job_json)


# modelop.metrics
def metrics(dataframe: pandas.DataFrame) -> dict:
    """A function to compute binary classification metrics and return record count given a sample (prod) dataset

    Args:
        dataframe (pandas.DataFrame): Sample (prod) dataset containing scores (model outputs)
        and labels (ground truths)

    Returns:
        dict: Combined binary classification metrics (accuracy, precision, recall, AUC, F1_score, confusion
        matrix) and record count
    """

    # Initialize ModelEvaluator
    model_evaluator = performance.ModelEvaluator(dataframe=dataframe, job_json=JOB)

    # Compute classification metrics
    performance_result = model_evaluator.evaluate_performance(
        pre_defined_metrics="classification_metrics"
    )

    # Initialize Volumetric monitor with input DataFrame
    volumetric_monitor = volumetrics.VolumetricMonitor(dataframe=dataframe)

    volumetric_result = volumetric_monitor.count(job_json=JOB)

    # Combine results into a single dict
    combined_result = {**performance_result, **volumetric_result}

    yield combined_result