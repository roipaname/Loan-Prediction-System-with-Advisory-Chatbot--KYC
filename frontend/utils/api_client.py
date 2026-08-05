"""
Thin requests-based client for the LAPAS backend. Read endpoints are cached
briefly with st.cache_data. get_applicants_safe() is what pages should
actually call — falls back to mock data if the backend's unreachable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 15


class BackendUnavailable(Exception):
    pass


def _get(path: str, **params: Any) -> Any:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise BackendUnavailable(str(exc)) from exc


def _post(path: str, json_body: Optional[dict] = None, timeout: int = 60, **params: Any) -> Any:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise BackendUnavailable(str(exc)) from exc


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_applicants(limit: int = 5000):
    return _get("/applicants", limit=limit)


def get_applicants(limit: int = 5000) -> pd.DataFrame:
    """Full applicant + features + latest-prediction dataset. Raises
    BackendUnavailable if unreachable — use get_applicants_safe() instead."""
    records = _fetch_applicants(limit=limit)
    df = pd.DataFrame(records)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_applicants_safe(limit: int = 5000) -> Tuple[pd.DataFrame, bool]:
    """Returns (df, used_mock_fallback)."""
    try:
        df = get_applicants(limit=limit)
        if df.empty:
            raise BackendUnavailable("Backend returned no applicants")
        return df, False
    except BackendUnavailable:
        from utils.mock_data import get_data
        return get_data(), True


def get_applicant_row(code: str) -> Optional[pd.Series]:
    df = get_applicants()
    match = df[df["id"] == code]
    return match.iloc[0] if not match.empty else None


def submit_application(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = _post("/applicants", json_body=payload)
    _fetch_applicants.clear()
    return result


@st.cache_data(ttl=30, show_spinner=False)
def get_prediction(code: str) -> Dict[str, Any]:
    return _get(f"/predictions/{code}")


def generate_advisory(code: str, retriever: str = "tfidf") -> Dict[str, Any]:
    # LLM call + retry can take a while, give it a longer timeout
    return _post(f"/predictions/{code}/advisory", retriever=retriever, timeout=100)


@st.cache_data(ttl=60, show_spinner=False)
def get_model_comparison() -> pd.DataFrame:
    return pd.DataFrame(_get("/comparisons/models"))


@st.cache_data(ttl=60, show_spinner=False)
def get_retrieval_comparison() -> pd.DataFrame:
    return pd.DataFrame(_get("/comparisons/retrieval"))


def backend_healthy() -> bool:
    try:
        requests.get(f"{API_BASE_URL}/health", timeout=2).raise_for_status()
        return True
    except requests.RequestException:
        return False
