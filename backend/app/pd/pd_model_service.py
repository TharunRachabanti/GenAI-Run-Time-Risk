"""
Fixed PD (Probability of Default) Model Service
Trains, validates, freezes, and applies the Logistic Regression PD model.
The PD score is the ONLY role of this ML model — it is a controlled input to GenAI experiments.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount_requested",
    "monthly_annuity",
    "debt_to_income_ratio",
    "credit_to_income_ratio",
    "employment_years",
    "credit_history_years",
    "delinquency_90day_count",
    "delinquency_30day_count",
    "external_credit_score_proxy",
    "combined_loan_to_value_ratio",
]

FEATURE_FILL_VALUES: Dict[str, float] = {
    "monthly_annuity": 0.0,
    "combined_loan_to_value_ratio": 80.0,
    "credit_to_income_ratio": 0.0,
    "employment_years": 0.0,
    "credit_history_years": 0.0,
    "delinquency_90day_count": 0.0,
    "delinquency_30day_count": 0.0,
    "external_credit_score_proxy": 0.5,
}

# ============================================================
# HOME CREDIT FEATURE MAPPING
# ============================================================
HOME_CREDIT_MAPPING = {
    "AMT_INCOME_TOTAL": "annual_income",
    "AMT_CREDIT": "loan_amount_requested",
    "AMT_ANNUITY": "monthly_annuity",
    "EXT_SOURCE_2": "external_credit_score_proxy",
}


def prepare_home_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Home Credit dataset columns to the research applicant schema.
    Apply all feature engineering and leakage prevention.

    IMPORTANT: This function handles the DAYS_EMPLOYED=365243 sentinel value
    which indicates 'not employed' in the Home Credit dataset.
    """
    out = pd.DataFrame()

    # Income
    out["annual_income"] = df["AMT_INCOME_TOTAL"].clip(lower=0)

    # Loan amount
    out["loan_amount_requested"] = df["AMT_CREDIT"].clip(lower=0)

    # Annuity (monthly payment)
    out["monthly_annuity"] = df["AMT_ANNUITY"].fillna(0).clip(lower=0)

    # Derived: DTI = monthly annuity / (annual income / 12)
    monthly_income = out["annual_income"] / 12
    out["debt_to_income_ratio"] = (
        out["monthly_annuity"] / monthly_income.replace(0, np.nan) * 100
    ).fillna(0).clip(0, 200)

    # Derived: Credit-to-income ratio
    out["credit_to_income_ratio"] = (
        out["loan_amount_requested"] / out["annual_income"].replace(0, np.nan)
    ).fillna(0).clip(0, 100)

    # Employment years (DAYS_EMPLOYED is negative; 365243 = not employed)
    days_employed = df["DAYS_EMPLOYED"].copy()
    not_employed_mask = days_employed == 365243
    days_employed = days_employed.abs()
    days_employed[not_employed_mask] = 0
    out["employment_years"] = (days_employed / 365.25).fillna(0).clip(lower=0)

    # Credit history (DAYS_REGISTRATION)
    if "DAYS_REGISTRATION" in df.columns:
        out["credit_history_years"] = (df["DAYS_REGISTRATION"].abs() / 365.25).fillna(0).clip(lower=0)
    else:
        out["credit_history_years"] = 0.0

    # Delinquency
    if "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns:
        out["delinquency_30day_count"] = df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0).clip(lower=0)
    else:
        out["delinquency_30day_count"] = 0.0

    if "DEF_60_CNT_SOCIAL_CIRCLE" in df.columns:
        out["delinquency_90day_count"] = df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0).clip(lower=0)
    else:
        out["delinquency_90day_count"] = 0.0

    # External credit score (higher = lower risk in Home Credit)
    out["external_credit_score_proxy"] = df["EXT_SOURCE_2"].fillna(0.5).clip(0, 1)

    # LTV proxy
    if "AMT_GOODS_PRICE" in df.columns:
        goods_price = df["AMT_GOODS_PRICE"].fillna(df["AMT_CREDIT"])
        out["combined_loan_to_value_ratio"] = (
            df["AMT_CREDIT"] / goods_price.replace(0, np.nan) * 100
        ).fillna(80).clip(0, 200)
    else:
        out["combined_loan_to_value_ratio"] = 80.0

    return out[NUMERIC_FEATURES]


# ============================================================
# PD MODEL CLASS
# ============================================================
class PDModelService:
    """
    Encapsulates the Logistic Regression PD model lifecycle.

    This model is NOT the main research subject.
    It converts application-time variables into a fixed PD score
    that serves as a controlled, frozen input to GenAI experiments.
    """

    MODEL_VERSION = "1.0.0"
    MODEL_NAME = "LR-PD-HomeCredit-v1"

    def __init__(self, model_dir: str = "./models/pd"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Optional[Pipeline] = None
        self._is_frozen: bool = False
        self._metadata: Dict[str, Any] = {}

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def pipeline(self) -> Optional[Pipeline]:
        return self._pipeline

    def build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline: StandardScaler → LogisticRegression."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ])

    def train_and_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train the Logistic Regression PD model with train/validation split.
        Returns validation metrics.
        """
        if self._is_frozen:
            raise RuntimeError("PD model is frozen. Cannot retrain.")

        logger.info("Starting PD model training...")

        # Fill missing values
        X_clean = X.copy()
        for col, fill_val in FEATURE_FILL_VALUES.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(fill_val)

        # Train/test split (stratified)
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")
        logger.info(f"Default rate (train): {y_train.mean():.3%}")
        logger.info(f"Default rate (val): {y_val.mean():.3%}")

        # Build and train pipeline
        self._pipeline = self.build_pipeline()
        self._pipeline.fit(X_train, y_train)

        # Predict
        y_pred = self._pipeline.predict(X_val)
        y_prob = self._pipeline.predict_proba(X_val)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        # KS statistic
        from scipy import stats
        ks_stat, _ = stats.ks_2samp(
            y_prob[y_val == 0],
            y_prob[y_val == 1],
        )

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)

        validation_results = {
            "training_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "default_rate_train": float(y_train.mean()),
            "default_rate_val": float(y_val.mean()),
            "roc_auc": float(roc_auc),
            "brier_score": float(brier),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "ks_statistic": float(ks_stat),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_val, y_pred, output_dict=True),
            "feature_list": NUMERIC_FEATURES,
            "model_version": self.MODEL_VERSION,
        }

        self._metadata = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "model_type": "LogisticRegression",
            "training_dataset": "home_credit_default_risk_v1",
            "training_dataset_version": "v1.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "validation_results": validation_results,
            "feature_list": NUMERIC_FEATURES,
        }

        logger.info(f"PD Model trained | ROC-AUC={roc_auc:.4f} | Brier={brier:.4f} | KS={ks_stat:.4f}")
        return validation_results

    def predict_pd(self, applicant_features: Dict[str, Any]) -> float:
        """
        Calculate PD score for a single applicant.
        Returns probability of default (0.0–1.0).
        """
        if self._pipeline is None:
            raise RuntimeError("PD model not trained or loaded.")

        # Build feature vector
        features = {}
        for feat in NUMERIC_FEATURES:
            val = applicant_features.get(feat, FEATURE_FILL_VALUES.get(feat, 0.0))
            features[feat] = val if val is not None else FEATURE_FILL_VALUES.get(feat, 0.0)

        X = pd.DataFrame([features])[NUMERIC_FEATURES]
        pd_score = float(self._pipeline.predict_proba(X)[0, 1])
        return round(pd_score, 6)

    def predict_pd_batch(self, applicant_list: List[Dict[str, Any]]) -> List[float]:
        """Calculate PD scores for a batch of applicants."""
        if self._pipeline is None:
            raise RuntimeError("PD model not trained or loaded.")

        rows = []
        for applicant in applicant_list:
            row = {}
            for feat in NUMERIC_FEATURES:
                val = applicant.get(feat, FEATURE_FILL_VALUES.get(feat, 0.0))
                row[feat] = val if val is not None else FEATURE_FILL_VALUES.get(feat, 0.0)
            rows.append(row)

        X = pd.DataFrame(rows)[NUMERIC_FEATURES]
        probs = self._pipeline.predict_proba(X)[:, 1]
        return [round(float(p), 6) for p in probs]

    def freeze(self) -> None:
        """
        FREEZE the PD model.
        After freezing:
        - No retraining is allowed
        - PD scores become immutable for all applicants
        """
        if self._pipeline is None:
            raise RuntimeError("Cannot freeze: model not trained.")
        self._is_frozen = True
        self._metadata["frozen_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("PD model FROZEN. No further training or score changes permitted.")

    def save(self, artifact_dir: Optional[str] = None) -> Dict[str, str]:
        """Save model artifacts and metadata."""
        if self._pipeline is None:
            raise RuntimeError("No model to save.")

        save_dir = Path(artifact_dir) if artifact_dir else self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"pd_model_{self.MODEL_VERSION}.joblib"
        metadata_path = save_dir / f"pd_model_{self.MODEL_VERSION}_metadata.json"

        joblib.dump(self._pipeline, model_path)

        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info(f"PD model saved to {model_path}")
        return {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        }

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a saved model pipeline."""
        load_path = Path(model_path) if model_path else (
            self.model_dir / f"pd_model_{self.MODEL_VERSION}.joblib"
        )

        if not load_path.exists():
            raise FileNotFoundError(f"PD model artifact not found: {load_path}")

        self._pipeline = joblib.load(load_path)
        self._is_frozen = True  # Loaded models are always treated as frozen
        logger.info(f"PD model loaded from {load_path} (treated as frozen)")

        # Try to load metadata
        metadata_path = load_path.parent / load_path.name.replace(".joblib", "_metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return absolute logistic regression coefficients as feature importance."""
        if self._pipeline is None:
            raise RuntimeError("Model not available.")
        coefs = self._pipeline.named_steps["classifier"].coef_[0]
        return {feat: float(abs(coef)) for feat, coef in zip(NUMERIC_FEATURES, coefs)}

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()
