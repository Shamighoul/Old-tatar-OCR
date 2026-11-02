"""Configurable image preprocessing (EXIF -> document -> perspective -> crop -> deskew -> resize -> contrast)."""

from preprocessing.pipeline import PreprocessConfig, PreprocessResult, load_preprocess_config, preprocess

__all__ = ["PreprocessConfig", "PreprocessResult", "load_preprocess_config", "preprocess"]
