import time
import logging
import cProfile
from contextlib import contextmanager
from functools import wraps
from typing import Generator, Iterable
import pandas as pd

# Basic logging configuration suitable for library use; projects can override root config.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def timer_decorator(func):
    """Decorator to log execution time of functions.

    Usage:
        @timer_decorator
        def foo(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.time() - start
            logger.info(f"{func.__name__} executed in {elapsed:.4f} seconds")
    return wrapper


@contextmanager
def profile_context(section_name: str):
    """Context manager for lightweight profiling using cProfile.

    Example:
        with profile_context("train_loop"):
            train()
    """
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        logger.info(f"Profiling stats for {section_name}:")
        profiler.print_stats(sort='cumtime')


def data_generator(df: pd.DataFrame, batch_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
    """Yield successive batches of a DataFrame for memory-efficient processing.

    Args:
        df: Input pandas DataFrame.
        batch_size: Number of rows per batch.
    Yields:
        DataFrame slices of up to batch_size rows.
    """
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield df.iloc[start:end]
