"""Thin entry-point shim — import CurationPipeline from pipeline.py instead."""

from policy_doctor.curation_pipeline.pipeline import ALL_STEPS, CurationPipeline

__all__ = ["ALL_STEPS", "CurationPipeline"]
