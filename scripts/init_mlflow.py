#!/usr/bin/env python3
"""
Simple MLflow Initializer

Quick script to initialize MLflow tracking URI for pipeline steps.
"""

import mlflow
import sys


def main():
    """Initialize MLflow tracking."""
    try:
        mlflow.set_tracking_uri('file:./mlruns')
        print("MLflow tracking URI set to file:./mlruns")
        return 0
    except Exception as e:
        print(f"Failed to initialize MLflow: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
