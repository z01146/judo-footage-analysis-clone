import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

DEFAULT_MEMORY_TO_CORE_RATIO = 0.75


def get_spark(
    cores=os.cpu_count(),
    memory=f"4g",
    local_dir="/tmp/spark",
    app_name=None,
    **kwargs,
):
    """Get a spark session for a single driver."""
    builder = (
        SparkSession.builder.config("spark.driver.cores", cores)
        .config("spark.driver.memory", memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", f"{local_dir}/{int(time.time())}")
    )
    for k, v in kwargs.items():
        builder = builder.config(k, v)
    if not app_name:
        app_name = f"spark-{int(time.time())}"
    return builder.appName(app_name).master(f"local[{cores}]").getOrCreate()


@contextmanager
def spark_resource(*args, **kwargs):
    """A context manager for a spark session."""
    spark = None
    try:
        spark = get_spark(*args, **kwargs)
        yield spark
    finally:
        if spark is not None:
            spark.stop()


def ensure_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path):
    path = Path(path)
    ensure_path(path.parent)
    return path
