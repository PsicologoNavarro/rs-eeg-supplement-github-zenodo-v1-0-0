from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests


REPRESENTATIONS = ["OC", "OA", "DELTA_OC_MINUS_OA"]
REPRESENTATION_CONDITIONS = {
    "OC": "eyes_closed",
    "OA": "eyes_open",
}
EEG_IDS = [f"EEG{i}" for i in range(1, 65)]
OUTCOME_MAP = {"BAI": "STAI_Trait_Anxiety", "MSPSS": "MSPSS_total"}
N_EXPECTED = 135
MIN_BINARY_GROUP = 5
MIN_CONTINUOUS_N = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validación LEMON de redes nominales locales y todas las W utilizables."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--lemon-repo",
        type=Path,
        default=Path.cwd() / "external_rseeg_transfer",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("lemon_nominal_validation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        output / "execution.log", encoding="utf-8", mode="w"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def zscore_sample(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std(ddof=1)


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def parse_numeric_text(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def age_midpoint(value: object) -> float:
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if len(numbers) < 2:
        return np.nan
    return (float(numbers[0]) + float(numbers[1])) / 2.0


def load_local_discovery(
    project: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    result_path = (
        project
        / "output_oa_oc_delta_comparison"
        / "resultados_1920_comparados.csv"
    )
    mapping_path = project / "output" / "eeg_variable_mapping.csv"
    moderator_path = project / "output" / "moderator_mapping.csv"
    for path in [result_path, mapping_path, moderator_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    all_local = pd.read_csv(result_path)
    selected = all_local.loc[
        all_local["status"].eq("ok") & (all_local["p_raw"] < 0.05)
    ].copy()
    if len(selected) != 51:
        raise ValueError(f"Se esperaban 51 filas locales nominales; hay {len(selected)}.")
    counts = selected.groupby("analysis_family").size().to_dict()
    if counts != {"DELTA_OC_MINUS_OA": 18, "OA": 2, "OC": 31}:
        raise ValueError(f"Conteos locales inesperados: {counts}")

    mapping = pd.read_csv(mapping_path)
    if len(mapping) != 64 or mapping["eeg_id"].tolist() != EEG_IDS:
        raise ValueError("Mapping EEG local inválido.")
    local_moderators = pd.read_csv(moderator_path)

    target_rows = []
    for keys, block in selected.groupby(
        ["analysis_family", "outcome", "eeg_id"], sort=False
    ):
        representation, outcome, eeg_id = keys
        first = block.iloc[0]
        target_rows.append(
            {
                "target_id": f"TARGET__{representation}__{outcome}__{eeg_id}",
                "representation": representation,
                "local_outcome": outcome,
                "lemon_outcome": OUTCOME_MAP[outcome],
                "eeg_id": eeg_id,
                "channel": first["channel"],
                "band": first["band"],
                "local_rows": len(block),
                "local_moderators": " | ".join(block["moderator_id"].astype(str)),
                "local_moderator_names": " | ".join(block["moderator"].astype(str)),
                "local_beta_interactions": " | ".join(
                    f"{value:.10g}" for value in block["beta_interaction"]
                ),
                "local_p_min": float(block["p_raw"].min()),
            }
        )
    targets = pd.DataFrame(target_rows)
    if len(targets) != 50:
        raise ValueError(f"Se esperaban 50 blancos únicos; hay {len(targets)}.")

    component_rows = []
    network_rows = []
    group_columns = ["analysis_family", "outcome", "moderator_id"]
    for index, (keys, block) in enumerate(
        selected.groupby(group_columns, sort=False), start=1
    ):
        representation, outcome, local_w = keys
        network_id = f"NET{index:02d}__{representation}__{outcome}__{local_w}"
        component_count = len(block)
        block = block.sort_values(["channel", "band", "eeg_id"]).copy()
        signs = np.sign(block["beta_interaction"].to_numpy(float))
        weights = signs / component_count
        for row, sign, weight in zip(
            block.itertuples(index=False), signs, weights
        ):
            component_rows.append(
                {
                    "network_id": network_id,
                    "representation": representation,
                    "local_outcome": outcome,
                    "lemon_outcome": OUTCOME_MAP[outcome],
                    "local_moderator_id": local_w,
                    "local_moderator": row.moderator,
                    "eeg_id": row.eeg_id,
                    "channel": row.channel,
                    "band": row.band,
                    "local_beta_interaction": row.beta_interaction,
                    "local_p_raw": row.p_raw,
                    "weight_sign_equal": weight,
                }
            )
        network_rows.append(
            {
                "network_id": network_id,
                "representation": representation,
                "local_outcome": outcome,
                "lemon_outcome": OUTCOME_MAP[outcome],
                "local_moderator_id": local_w,
                "local_moderator": block.iloc[0]["moderator"],
                "local_moderator_label": block.iloc[0]["moderator_label"],
                "n_components": component_count,
                "local_min_p": float(block["p_raw"].min()),
                "local_max_abs_beta": float(
                    block["beta_interaction"].abs().max()
                ),
                "eeg_ids": " | ".join(block["eeg_id"]),
            }
        )
    components = pd.DataFrame(component_rows)
    networks = pd.DataFrame(network_rows)
    if len(networks) != 8 or len(components) != 51:
        raise ValueError("Reconstrucción de redes locales inválida.")

    hashes = {
        str(path): sha256_file(path)
        for path in [result_path, mapping_path, moderator_path]
    }
    return selected, targets, networks, components, hashes


def load_lemon_data(
    lemon_repo: Path,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    status_path = lemon_repo / "outputs" / "06_features" / "feature_extraction_status.csv"
    eligible_path = lemon_repo / "outputs" / "02_cohort" / "lemon_eligible_subjects.csv"
    eeg_path = (
        lemon_repo
        / "outputs"
        / "06_features"
        / "participant_condition_bandpower.csv"
    )
    metadata_dir = lemon_repo / "data" / "lemon" / "metadata"
    meta_path = (
        metadata_dir
        / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    )
    mspss_path = metadata_dir / "MSPSS.csv"
    stai_path = metadata_dir / "STAI_G_X2.csv"
    paths = [status_path, eligible_path, eeg_path, meta_path, mspss_path, stai_path]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    status = pd.read_csv(status_path)
    eligible = pd.read_csv(eligible_path)
    pass_ids = set(status.loc[status["status"].eq("PASS"), "participant_id"])
    cohort = eligible.loc[eligible["ID"].isin(pass_ids)].copy()
    cohort = cohort.dropna(subset=["STAI_Trait_Anxiety", "MSPSS_total"])
    cohort = cohort.sort_values("ID").reset_index(drop=True)
    if len(cohort) != N_EXPECTED or cohort["ID"].nunique() != N_EXPECTED:
        raise ValueError(f"Cohorte LEMON esperada n=135; observada n={len(cohort)}.")

    meta = pd.read_csv(meta_path)
    mspss = pd.read_csv(mspss_path)
    stai = pd.read_csv(stai_path)
    data = (
        cohort[["ID", "Age", "Gender_ 1=female_2=male", "STAI_Trait_Anxiety", "MSPSS_total"]]
        .merge(meta, on="ID", how="left", suffixes=("", "_meta"))
        .merge(mspss, on="ID", how="left", suffixes=("", "_scale"))
        .merge(stai, on="ID", how="left", suffixes=("", "_stai"))
    )
    data["STAI_Trait_Anxiety"] = numeric_series(data["STAI_Trait_Anxiety"])
    data["MSPSS_total"] = numeric_series(data["MSPSS_total"])

    eeg = pd.read_csv(eeg_path)
    eeg = eeg.loc[
        eeg["cohort"].eq("lemon") & eeg["participant_id"].isin(data["ID"])
    ].copy()
    expected_rows = N_EXPECTED * 2 * 64
    if len(eeg) != expected_rows:
        raise ValueError(f"EEG LEMON: {len(eeg)} filas, esperadas {expected_rows}.")
    if eeg["participant_median_log_power"].isna().any():
        raise ValueError("Hay potencia EEG faltante.")
    duplicate_key = ["participant_id", "condition", "channel", "band"]
    if eeg.duplicated(duplicate_key).any():
        raise ValueError("Hay celdas EEG duplicadas.")
    pivot = eeg.pivot(
        index="participant_id",
        columns=["condition", "channel", "band"],
        values="participant_median_log_power",
    ).reindex(data["ID"])

    representations: dict[str, pd.DataFrame] = {}
    for representation, condition in REPRESENTATION_CONDITIONS.items():
        matrix = pd.DataFrame(index=data["ID"], columns=EEG_IDS, dtype=float)
        for row in mapping.itertuples(index=False):
            matrix[row.eeg_id] = pivot[(condition, row.channel, row.band)].to_numpy(float)
        representations[representation] = matrix
    representations["DELTA_OC_MINUS_OA"] = (
        representations["OC"] - representations["OA"]
    )
    for name, matrix in representations.items():
        if matrix.shape != (N_EXPECTED, 64) or matrix.isna().any().any():
            raise ValueError(f"Matriz EEG {name} inválida: {matrix.shape}.")

    hashes = {str(path): sha256_file(path) for path in paths}
    return data, representations, eeg, hashes


def build_lemon_moderators(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = pd.DataFrame({"ID": data["ID"]})
    definitions: list[dict] = []

    def add(
        name: str,
        series: pd.Series,
        kind: str,
        label: str,
        source: str,
        positive: str,
        transform: str,
        local_analogue: str,
        equivalence: str,
    ) -> None:
        values[name] = pd.to_numeric(series, errors="coerce")
        definitions.append(
            {
                "moderator": name,
                "type": kind,
                "label": label,
                "source_column": source,
                "positive_or_high_definition": positive,
                "transform": transform,
                "local_analogue": local_analogue,
                "equivalence": equivalence,
            }
        )

    gender = numeric_series(data["Gender_ 1=female_2=male"])
    add(
        "GENDER_FEMALE",
        gender.map({1.0: 1.0, 2.0: 0.0}),
        "binary",
        "Mujer vs hombre",
        "Gender_ 1=female_2=male",
        "1=mujer",
        "native_binary",
        "W1 SEX_FEMALE",
        "direct",
    )
    add(
        "AGE_MIDPOINT",
        data["Age"].map(age_midpoint),
        "continuous",
        "Edad por punto medio del intervalo",
        "Age",
        "mayor edad",
        "interval_midpoint_then_z",
        "none",
        "new_W",
    )
    handed = data["Handedness"].astype("string").str.strip().str.lower()
    add(
        "HANDEDNESS_NONRIGHT",
        pd.Series(
            np.where(handed.eq("right"), 0.0, np.where(handed.notna(), 1.0, np.nan)),
            index=data.index,
        ),
        "binary",
        "No diestro vs diestro",
        "Handedness",
        "1=izquierdo/ambidiestro",
        "derived_binary",
        "none",
        "new_W",
    )
    education = data["Education"].astype("string").str.strip().str.lower()
    high_education = education.isin(["gymnasium", "gymansium"])
    valid_education = high_education | education.str.contains(
        "realschule|hauptschule|none", regex=True, na=False
    )
    education_lower = pd.Series(np.nan, index=data.index)
    education_lower.loc[valid_education] = (~high_education.loc[valid_education]).astype(float)
    add(
        "EDUCATION_LOWER",
        education_lower,
        "binary",
        "Escolaridad menor que Gymnasium",
        "Education",
        "1=Realschule/Hauptschule/incompleta",
        "correct_typo_then_binary",
        "none",
        "new_W",
    )
    add(
        "DRUG_SCREEN_POSITIVE",
        numeric_series(data["DRUG_0=negative_1=Positive"]),
        "binary",
        "Tamiz de drogas positivo",
        "DRUG_0=negative_1=Positive",
        "1=positivo",
        "native_binary",
        "none",
        "new_W",
    )
    smoking = numeric_series(
        data["Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)"]
    )
    add(
        "SMOKING_ANY",
        pd.Series(
            np.where(smoking.notna(), (smoking > 1).astype(float), np.nan),
            index=data.index,
        ),
        "binary",
        "Fumador ocasional/regular",
        "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)",
        "1=ocasional o regular",
        "derived_binary",
        "none",
        "new_W",
    )
    skid = data["SKID_Diagnoses"].astype("string").str.strip().str.lower()
    add(
        "ANY_SKID_DIAGNOSIS",
        pd.Series(
            np.where(
                skid.isna(),
                np.nan,
                np.where(skid.str.fullmatch(r"none\s*", na=False), 0.0, 1.0),
            ),
            index=data.index,
        ),
        "binary",
        "Cualquier diagnóstico SKID",
        "SKID_Diagnoses",
        "1=diagnóstico presente/pasado",
        "none_vs_any",
        "none",
        "new_W",
    )
    hamilton = numeric_series(data["Hamilton_Scale"])
    add(
        "HAMILTON_LOG1P",
        np.log1p(hamilton),
        "continuous",
        "Escala Hamilton",
        "Hamilton_Scale",
        "mayor sintomatología",
        "log1p_then_z",
        "none",
        "new_W_clinical",
    )
    bsl_sum = numeric_series(data["BSL23_sumscore"])
    add(
        "BSL23_SUM_LOG1P",
        np.log1p(bsl_sum),
        "continuous",
        "BSL-23 suma",
        "BSL23_sumscore",
        "mayor sintomatología",
        "log1p_then_z",
        "none",
        "new_W_clinical",
    )
    bsl_behavior = numeric_series(data["BSL23_behavior"])
    add(
        "BSL23_BEHAVIOR_ANY",
        pd.Series(
            np.where(
                bsl_behavior.notna(), (bsl_behavior > 0).astype(float), np.nan
            ),
            index=data.index,
        ),
        "binary",
        "Alguna conducta BSL-23",
        "BSL23_behavior",
        "1=cualquier conducta",
        "zero_vs_positive",
        "none",
        "new_W_clinical",
    )
    audit_score = numeric_series(data["AUDIT"])
    add(
        "AUDIT_LOG1P",
        np.log1p(audit_score),
        "continuous",
        "AUDIT",
        "AUDIT",
        "mayor riesgo por alcohol",
        "log1p_then_z",
        "none",
        "new_W_clinical",
    )
    alcohol_units = data["Standard_Alcoholunits_Last_28days"].map(
        parse_numeric_text
    )
    add(
        "ALCOHOL_UNITS_LOG1P",
        np.log1p(alcohol_units),
        "continuous",
        "Unidades estándar de alcohol en 28 días",
        "Standard_Alcoholunits_Last_28days",
        "mayor consumo",
        "parse_numeric_log1p_then_z",
        "none",
        "new_W_clinical",
    )
    family_alcohol = (
        data["Alcohol_Dependence_In_1st-3rd_Degree_relative"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    family_alcohol_binary = pd.Series(np.nan, index=data.index)
    family_alcohol_binary.loc[family_alcohol.eq("no")] = 0.0
    family_alcohol_binary.loc[family_alcohol.str.startswith("yes", na=False)] = 1.0
    add(
        "FAMILY_ALCOHOL_DEPENDENCE",
        family_alcohol_binary,
        "binary",
        "Dependencia de alcohol en familiar",
        "Alcohol_Dependence_In_1st-3rd_Degree_relative",
        "1=sí",
        "yes_variants_vs_no_unknown_missing",
        "none",
        "new_W",
    )
    relationship = (
        data["Relationship_Status"].astype("string").str.strip().str.lower()
    )
    relationship_binary = pd.Series(np.nan, index=data.index)
    relationship_binary.loc[relationship.eq("yes")] = 0.0
    relationship_binary.loc[relationship.eq("no")] = 1.0
    add(
        "RELATIONSHIP_NO_PARTNER",
        relationship_binary,
        "binary",
        "Sin pareja vs con pareja",
        "Relationship_Status",
        "1=sin pareja",
        "reverse_native_binary",
        "W4 LIVING_ALONE",
        "conceptual_not_equivalent",
    )

    definition_table = pd.DataFrame(definitions)
    quality_rows = []
    retained = []
    for row in definition_table.itertuples(index=False):
        series = values[row.moderator]
        finite = series.dropna()
        quality = {
            "moderator": row.moderator,
            "type": row.type,
            "n": len(finite),
            "missing": int(series.isna().sum()),
            "unique": int(finite.nunique()),
            "minimum": float(finite.min()) if len(finite) else np.nan,
            "maximum": float(finite.max()) if len(finite) else np.nan,
            "n_0": int((finite == 0).sum()) if row.type == "binary" else np.nan,
            "n_1": int((finite == 1).sum()) if row.type == "binary" else np.nan,
        }
        if row.type == "binary":
            valid = (
                set(finite.unique()).issubset({0.0, 1.0})
                and min(quality["n_0"], quality["n_1"]) >= MIN_BINARY_GROUP
            )
            reason = "retained" if valid else "binary_group_below_5_or_invalid"
        else:
            valid = len(finite) >= MIN_CONTINUOUS_N and finite.nunique() >= 3
            reason = "retained" if valid else "continuous_n_below_100_or_below_3_levels"
        quality["retained"] = bool(valid)
        quality["decision"] = reason
        quality_rows.append(quality)
        if valid:
            retained.append(row.moderator)
    quality_table = pd.DataFrame(quality_rows)
    definitions = definition_table.merge(quality_table, on=["moderator", "type"])
    definitions = definitions.loc[definitions["retained"]].reset_index(drop=True)
    values = values[["ID", *retained]]
    if len(definitions) != 14:
        raise ValueError(
            f"Se esperaban 14 moderadores utilizables; hay {len(definitions)}: {retained}"
        )
    return values, definitions, quality_table


def build_variable_inventory(
    data: pd.DataFrame,
    moderator_definitions: pd.DataFrame,
) -> pd.DataFrame:
    decisions = {
        "Gender_ 1=female_2=male": ("derived", "GENDER_FEMALE"),
        "Age": ("derived", "AGE_MIDPOINT"),
        "Handedness": ("derived", "HANDEDNESS_NONRIGHT"),
        "Education": ("derived", "EDUCATION_LOWER"),
        "DRUG": ("duplicate_text", "Use numeric drug screen"),
        "DRUG_0=negative_1=Positive": ("derived", "DRUG_SCREEN_POSITIVE"),
        "Unnamed: 7": ("excluded_sparse", "Only substance detail; very sparse"),
        "Smoking": ("duplicate_text", "Use numeric smoking source"),
        "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)": (
            "derived",
            "SMOKING_ANY",
        ),
        "SKID_Diagnoses": ("derived", "ANY_SKID_DIAGNOSIS"),
        "SKID_Diagnoses 1": ("duplicate_coded_text", "Same SKID construct"),
        "SKID_Diagnoses 2": ("excluded_sparse", "Only four nonmissing"),
        "Comments_SKID_assessment": ("excluded_free_text", "No stable coding"),
        "Hamilton_Scale": ("derived", "HAMILTON_LOG1P"),
        "BSL23_sumscore": ("derived", "BSL23_SUM_LOG1P"),
        "BSL23_behavior": ("derived", "BSL23_BEHAVIOR_ANY"),
        "AUDIT": ("derived", "AUDIT_LOG1P"),
        "Standard_Alcoholunits_Last_28days": ("derived", "ALCOHOL_UNITS_LOG1P"),
        "Alcohol_Dependence_In_1st-3rd_Degree_relative": (
            "derived",
            "FAMILY_ALCOHOL_DEPENDENCE",
        ),
        "Relationship_Status": ("derived", "RELATIONSHIP_NO_PARTNER"),
        "MSPSS_SignificantOthers": (
            "excluded_construct_leakage",
            "Component of MSPSS total; not W",
        ),
        "MSPSS_Family": (
            "excluded_construct_leakage",
            "Component of MSPSS total; not W",
        ),
        "MSPSS_Friends": (
            "excluded_construct_leakage",
            "Component of MSPSS total; not W",
        ),
        "MSPSS_total": ("outcome", "External outcome for local MSPSS"),
        "STAI_Trait_Anxiety": ("outcome", "Conceptual proxy for local BAI"),
        "ID": ("identifier", "Merge key only"),
    }
    rows = []
    for column in decisions:
        if column not in data.columns:
            continue
        series = data[column]
        nonmissing = series.dropna()
        action, destination = decisions[column]
        rows.append(
            {
                "source_variable": column,
                "n_nonmissing_in_135": len(nonmissing),
                "unique_nonmissing": int(nonmissing.astype(str).nunique()),
                "action": action,
                "derived_or_reason": destination,
            }
        )
    inventory = pd.DataFrame(rows)
    used_sources = set(moderator_definitions["source_column"])
    inventory["used_as_W"] = inventory["source_variable"].isin(used_sources)
    return inventory


def construct_network_scores(
    representations: dict[str, pd.DataFrame],
    networks: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=next(iter(representations.values())).index)
    for network in networks.itertuples(index=False):
        block = components.loc[components["network_id"].eq(network.network_id)]
        matrix = representations[network.representation][block["eeg_id"]].to_numpy(float)
        standardized = np.column_stack(
            [zscore_sample(matrix[:, index]) for index in range(matrix.shape[1])]
        )
        weights = block["weight_sign_equal"].to_numpy(float)
        score = standardized @ weights
        if not np.isfinite(score).all() or np.std(score, ddof=1) <= 0:
            raise ValueError(f"Firma inválida: {network.network_id}")
        scores[network.network_id] = score
    return scores


def ols_interaction(
    y_raw: np.ndarray,
    x_raw: np.ndarray,
    w_raw: np.ndarray,
    w_type: str,
) -> dict:
    mask = np.isfinite(y_raw) & np.isfinite(x_raw) & np.isfinite(w_raw)
    y = zscore_sample(y_raw[mask])
    x = zscore_sample(x_raw[mask])
    w_base = w_raw[mask].astype(float)
    w = w_base if w_type == "binary" else zscore_sample(w_base)
    w_centered = w - w.mean()
    interaction = x * w_centered
    reduced = np.column_stack([np.ones(len(y)), x, w_centered])
    full = np.column_stack([reduced, interaction])
    if np.linalg.matrix_rank(full) != 4:
        raise ValueError("Diseño sin rango completo.")
    inverse = np.linalg.inv(full.T @ full)
    beta = inverse @ full.T @ y
    fitted = full @ beta
    residual = y - fitted
    df = len(y) - 4
    sse = float(residual @ residual)
    mse = sse / df
    se = np.sqrt(np.diag(inverse) * mse)
    t_value = beta[3] / se[3]
    p_value = 2 * stats.t.sf(abs(t_value), df)

    leverage = np.sum((full @ inverse) * full, axis=1)
    adjusted = residual / np.maximum(1.0 - leverage, 1e-12)
    meat = full.T @ ((adjusted[:, None] ** 2) * full)
    covariance_hc3 = inverse @ meat @ inverse
    se_hc3 = math.sqrt(max(covariance_hc3[3, 3], 0.0))
    t_hc3 = beta[3] / se_hc3
    p_hc3 = 2 * stats.t.sf(abs(t_hc3), df)

    beta_reduced = np.linalg.lstsq(reduced, y, rcond=None)[0]
    residual_reduced = y - reduced @ beta_reduced
    sse_reduced = float(residual_reduced @ residual_reduced)
    total = float(np.sum((y - y.mean()) ** 2))
    r2_full = 1.0 - sse / total
    r2_reduced = 1.0 - sse_reduced / total
    delta_r2 = r2_full - r2_reduced
    f2 = delta_r2 / max(1.0 - r2_full, np.finfo(float).eps)

    inverse_reduced = np.linalg.inv(reduced.T @ reduced)
    z_residual = interaction - reduced @ (
        inverse_reduced @ reduced.T @ interaction
    )
    z_norm2 = float(z_residual @ z_residual)
    reduced_residual_norm2 = float(residual_reduced @ residual_reduced)
    partial_r = float(
        residual_reduced @ z_residual
        / math.sqrt(reduced_residual_norm2 * z_norm2)
    )
    t_partial = partial_r * math.sqrt(df / max(1.0 - partial_r**2, 1e-15))
    if not np.isclose(t_partial, t_value, atol=1e-9):
        raise RuntimeError("La identidad de regresión parcial no coincide.")

    if w_type == "binary":
        mean_w = float(w_base.mean())
        slope_low = beta[1] + beta[3] * (0.0 - mean_w)
        slope_high = beta[1] + beta[3] * (1.0 - mean_w)
    else:
        slope_low = beta[1] - beta[3]
        slope_high = beta[1] + beta[3]
    return {
        "mask": mask,
        "n": int(mask.sum()),
        "df_resid": df,
        "beta_interaction_std": float(beta[3]),
        "se_interaction": float(se[3]),
        "t_interaction": float(t_value),
        "p_raw": float(p_value),
        "se_hc3": float(se_hc3),
        "t_hc3": float(t_hc3),
        "p_hc3": float(p_hc3),
        "r2_full": float(r2_full),
        "r2_reduced": float(r2_reduced),
        "delta_r2": float(delta_r2),
        "cohen_f2_interaction": float(f2),
        "slope_W_low_std": float(slope_low),
        "slope_W_high_std": float(slope_high),
        "condition_number": float(np.linalg.cond(full)),
        "reduced": reduced,
        "reduced_inverse": inverse_reduced,
        "reduced_residual": residual_reduced,
        "z_residual": z_residual,
        "z_norm2": z_norm2,
    }


def freedman_lane_t(
    fit: dict,
    permutation_orders: np.ndarray,
) -> np.ndarray:
    residual = fit["reduced_residual"]
    permuted_residual = residual[permutation_orders]
    reduced = fit["reduced"]
    inverse = fit["reduced_inverse"]
    projections = permuted_residual @ reduced
    q_norm2 = np.sum(permuted_residual**2, axis=1) - np.sum(
        (projections @ inverse) * projections, axis=1
    )
    q_norm2 = np.maximum(q_norm2, np.finfo(float).tiny)
    numerator = permuted_residual @ fit["z_residual"]
    denominator = np.sqrt(q_norm2 * fit["z_norm2"])
    partial_r = np.clip(numerator / denominator, -1 + 1e-12, 1 - 1e-12)
    return partial_r * np.sqrt(
        fit["df_resid"] / np.maximum(1.0 - partial_r**2, 1e-15)
    )


def add_interaction_inference(
    results: pd.DataFrame,
    null_abs_t: np.ndarray,
    target_column: str,
) -> pd.DataFrame:
    output = results.copy().reset_index(drop=True)
    observed = output["t_interaction"].abs().to_numpy(float)
    exceedances = np.sum(null_abs_t >= observed[None, :] - 1e-12, axis=0)
    output["permutation_exceedances"] = exceedances
    output["p_permutation"] = (exceedances + 1) / (len(null_abs_t) + 1)
    global_max = null_abs_t.max(axis=1)
    output["p_maxT_global"] = [
        (np.count_nonzero(global_max >= value - 1e-12) + 1)
        / (len(null_abs_t) + 1)
        for value in observed
    ]
    output["q_BH_raw_global"] = multipletests(
        output["p_raw"], method="fdr_bh"
    )[1]
    output["q_BH_HC3_global"] = multipletests(
        output["p_hc3"], method="fdr_bh"
    )[1]
    output["q_BH_permutation_global"] = multipletests(
        output["p_permutation"], method="fdr_bh"
    )[1]
    output["q_BY_permutation_global"] = multipletests(
        output["p_permutation"], method="fdr_by"
    )[1]
    output["q_BH_permutation_within_W"] = np.nan
    output[f"q_BH_permutation_within_{target_column}"] = np.nan
    output["p_maxT_within_W"] = np.nan
    for _, indices in output.groupby("external_moderator", sort=False).groups.items():
        idx = np.asarray(indices)
        output.loc[idx, "q_BH_permutation_within_W"] = multipletests(
            output.loc[idx, "p_permutation"], method="fdr_bh"
        )[1]
        local_max = null_abs_t[:, idx].max(axis=1)
        output.loc[idx, "p_maxT_within_W"] = [
            (np.count_nonzero(local_max >= value - 1e-12) + 1)
            / (len(null_abs_t) + 1)
            for value in observed[idx]
        ]
    for _, indices in output.groupby(target_column, sort=False).groups.items():
        idx = np.asarray(indices)
        output.loc[idx, f"q_BH_permutation_within_{target_column}"] = multipletests(
            output.loc[idx, "p_permutation"], method="fdr_bh"
        )[1]
    output["naive_p_lt_0_05"] = output["p_raw"] < 0.05
    output["hc3_p_lt_0_05"] = output["p_hc3"] < 0.05
    output["permutation_p_lt_0_05"] = output["p_permutation"] < 0.05
    output["robust_nominal"] = (
        output["naive_p_lt_0_05"] & output["permutation_p_lt_0_05"]
    )
    output["triangulated_nominal"] = (
        output["robust_nominal"] & output["hc3_p_lt_0_05"]
    )
    return output


def run_interaction_models(
    model_kind: str,
    definitions: pd.DataFrame,
    moderator_values: pd.DataFrame,
    data: pd.DataFrame,
    x_table: pd.DataFrame,
    x_metadata: pd.DataFrame,
    target_column: str,
    global_random_keys: np.ndarray,
    logger: logging.Logger,
    progress_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict] = []
    null_columns: list[np.ndarray] = []
    moderator_lookup = definitions.set_index("moderator")
    y_lookup = {
        name: data[name].to_numpy(float)
        for name in OUTCOME_MAP.values()
    }
    total = len(x_metadata) * len(definitions)
    completed = 0
    for moderator in definitions.itertuples(index=False):
        w_all = moderator_values[moderator.moderator].to_numpy(float)
        moderator_mask = np.isfinite(w_all)
        subset_keys = global_random_keys[:, moderator_mask]
        permutation_orders = np.argsort(
            subset_keys, axis=1, kind="stable"
        ).astype(np.int16)
        for target in x_metadata.itertuples(index=False):
            y_all = y_lookup[target.lemon_outcome]
            x_name = getattr(target, target_column)
            x_all = x_table[x_name].to_numpy(float)
            fit = ols_interaction(y_all, x_all, w_all, moderator.type)
            if not np.array_equal(fit["mask"], moderator_mask):
                raise RuntimeError("La máscara del modelo no coincide con la W.")
            null_t = freedman_lane_t(fit, permutation_orders)
            null_columns.append(np.abs(null_t).astype(np.float32))
            row = {
                "model_kind": model_kind,
                target_column: getattr(target, target_column),
                "representation": target.representation,
                "local_outcome": target.local_outcome,
                "lemon_outcome": target.lemon_outcome,
                "external_moderator": moderator.moderator,
                "external_moderator_label": moderator.label,
                "external_moderator_type": moderator.type,
                "external_W_positive_definition": moderator.positive_or_high_definition,
                "local_analogue": moderator.local_analogue,
                "W_equivalence": moderator.equivalence,
                "external_W_n_available": moderator.n,
                "external_W_missing": moderator.missing,
                "external_W_n0": moderator.n_0,
                "external_W_n1": moderator.n_1,
                **{
                    key: value
                    for key, value in fit.items()
                    if key
                    not in {
                        "mask",
                        "reduced",
                        "reduced_inverse",
                        "reduced_residual",
                        "z_residual",
                        "z_norm2",
                    }
                },
            }
            if model_kind == "network":
                row.update(
                    {
                        "local_moderator_id": target.local_moderator_id,
                        "local_moderator": target.local_moderator,
                        "n_components": target.n_components,
                        "local_min_p": target.local_min_p,
                    }
                )
            else:
                row.update(
                    {
                        "eeg_id": target.eeg_id,
                        "channel": target.channel,
                        "band": target.band,
                        "local_moderators": target.local_moderators,
                        "local_p_min": target.local_p_min,
                    }
                )
            rows.append(row)
            completed += 1
            if completed % 100 == 0 or completed == total:
                atomic_json(
                    progress_path,
                    {
                        "status": "running",
                        "stage": f"{model_kind}_interactions",
                        "completed": completed,
                        "total": total,
                        "percent": completed / total * 100,
                        "updated_at": datetime.now().astimezone().isoformat(),
                    },
                )
        logger.info(
            "%s: W %s completada (%d modelos acumulados).",
            model_kind,
            moderator.moderator,
            completed,
        )
    null_matrix = np.column_stack(null_columns)
    result = add_interaction_inference(
        pd.DataFrame(rows), null_matrix, target_column
    )
    return result, null_matrix


def main_effects(
    model_kind: str,
    data: pd.DataFrame,
    x_table: pd.DataFrame,
    metadata: pd.DataFrame,
    target_column: str,
    global_orders: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    null_columns = []
    for target in metadata.itertuples(index=False):
        x_name = getattr(target, target_column)
        x = zscore_sample(x_table[x_name].to_numpy(float))
        y = zscore_sample(data[target.lemon_outcome].to_numpy(float))
        r_value = float(x @ y / (len(y) - 1))
        p_raw = float(stats.pearsonr(x, y).pvalue)
        null_r = np.abs(y[global_orders] @ x / (len(y) - 1))
        null_columns.append(null_r.astype(np.float32))
        rows.append(
            {
                "model_kind": model_kind,
                target_column: getattr(target, target_column),
                "representation": target.representation,
                "local_outcome": target.local_outcome,
                "lemon_outcome": target.lemon_outcome,
                "n": len(y),
                "r_pearson": r_value,
                "p_raw": p_raw,
            }
        )
    null = np.column_stack(null_columns)
    output = pd.DataFrame(rows)
    observed = output["r_pearson"].abs().to_numpy(float)
    exceedances = np.sum(null >= observed[None, :] - 1e-12, axis=0)
    output["p_permutation"] = (exceedances + 1) / (len(null) + 1)
    output["q_BH_permutation"] = multipletests(
        output["p_permutation"], method="fdr_bh"
    )[1]
    max_null = null.max(axis=1)
    output["p_maxT"] = [
        (np.count_nonzero(max_null >= value - 1e-12) + 1) / (len(null) + 1)
        for value in observed
    ]
    return output, null


def conceptual_w4_subset(network_results: pd.DataFrame) -> pd.DataFrame:
    subset = network_results.loc[
        network_results["local_moderator_id"].eq("W4")
        & network_results["external_moderator"].eq("RELATIONSHIP_NO_PARTNER")
    ].copy()
    if len(subset):
        subset["q_BH_permutation_conceptual_W4"] = multipletests(
            subset["p_permutation"], method="fdr_bh"
        )[1]
    return subset


def plot_network_heatmap(
    results: pd.DataFrame,
    networks: pd.DataFrame,
    definitions: pd.DataFrame,
    output: Path,
) -> None:
    row_order = networks["network_id"].tolist()
    column_order = definitions["moderator"].tolist()
    pivot = (
        results.pivot(
            index="network_id",
            columns="external_moderator",
            values="beta_interaction_std",
        )
        .reindex(index=row_order, columns=column_order)
    )
    figure, axis = plt.subplots(
        figsize=(19, max(6, len(row_order) * 0.85)),
        constrained_layout=True,
    )
    limit = max(float(np.nanmax(np.abs(pivot.to_numpy()))), 0.25)
    image = axis.imshow(
        pivot.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axis.set_xticks(range(len(column_order)), column_order, rotation=55, ha="right")
    labels = []
    network_lookup = networks.set_index("network_id")
    for network_id in row_order:
        row = network_lookup.loc[network_id]
        labels.append(
            f"{network_id.split('__')[0]} | {row['representation']} | "
            f"{row['local_outcome']} | {row['local_moderator_id']} | "
            f"k={row['n_components']}"
        )
    axis.set_yticks(range(len(row_order)), labels)
    axis.set_title(
        "LEMON: interacción estandarizada firma EEG × W\n"
        "* p ingenua y permutación < .05; ‡ además HC3 < .05; "
        "† q BH global < .10"
    )
    for row_index, network_id in enumerate(row_order):
        for column_index, moderator in enumerate(column_order):
            match = results.loc[
                results["network_id"].eq(network_id)
                & results["external_moderator"].eq(moderator)
            ].iloc[0]
            marker = ""
            if match["q_BH_permutation_global"] < 0.10:
                marker = "†"
            elif match["triangulated_nominal"]:
                marker = "‡"
            elif match["robust_nominal"]:
                marker = "*"
            value = float(match["beta_interaction_std"])
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > limit * 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="β interacción estandarizada")
    figure.savefig(output / "heatmap_redes_por_W_LEMON.png", dpi=180)
    plt.close(figure)


def plot_main_effects(
    network_main: pd.DataFrame,
    feature_main: pd.DataFrame,
    output: Path,
) -> None:
    network_plot = network_main.sort_values("r_pearson")
    feature_plot = feature_main.sort_values("p_permutation").head(15)
    figure, axes = plt.subplots(
        1, 2, figsize=(18, 9), constrained_layout=True
    )
    colors_network = np.where(
        network_plot["r_pearson"] >= 0, "#C2410C", "#1D4ED8"
    )
    axes[0].barh(
        network_plot["network_id"],
        network_plot["r_pearson"],
        color=colors_network,
        alpha=0.82,
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("r de Pearson")
    axes[0].set_title(
        "Efectos principales: 8 firmas\n"
        "* q BH<.10; † maxT<.05"
    )
    for position, row in enumerate(network_plot.itertuples(index=False)):
        marker = "†" if row.p_maxT < 0.05 else ("*" if row.q_BH_permutation < 0.10 else "")
        axes[0].text(
            row.r_pearson,
            position,
            f" {row.r_pearson:.2f}{marker}",
            va="center",
            ha="left" if row.r_pearson >= 0 else "right",
            fontsize=9,
        )

    labels = [
        target.replace("TARGET__", "").replace("__", " | ")
        for target in feature_plot["target_id"]
    ]
    positions = np.arange(len(feature_plot))
    colors_feature = np.where(
        feature_plot["r_pearson"] >= 0, "#C2410C", "#1D4ED8"
    )
    axes[1].barh(
        positions,
        feature_plot["r_pearson"],
        color=colors_feature,
        alpha=0.82,
    )
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("r de Pearson")
    axes[1].set_title(
        "15 blancos canal×banda con menor p perm\n"
        "* q BH<.10; † maxT<.05"
    )
    for position, row in enumerate(feature_plot.itertuples(index=False)):
        marker = "†" if row.p_maxT < 0.05 else ("*" if row.q_BH_permutation < 0.10 else "")
        axes[1].text(
            row.r_pearson,
            position,
            f" {row.r_pearson:.2f}{marker}",
            va="center",
            ha="left" if row.r_pearson >= 0 else "right",
            fontsize=8,
        )
    figure.suptitle(
        "Apoyo externo EEG–outcome en LEMON\n"
        "(asociación principal; no réplica de moderación)",
        fontsize=15,
    )
    figure.savefig(output / "efectos_principales_externos.png", dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    local_selected: pd.DataFrame,
    targets: pd.DataFrame,
    networks: pd.DataFrame,
    definitions: pd.DataFrame,
    network_results: pd.DataFrame,
    feature_results: pd.DataFrame,
    network_main: pd.DataFrame,
    feature_main: pd.DataFrame,
    conceptual_w4: pd.DataFrame,
    permutations: int,
    seed: int,
) -> None:
    top_network = network_results.sort_values("p_permutation").head(15)
    top_feature = feature_results.sort_values("p_permutation").head(15)
    lines = [
        "VALIDACIÓN LEMON DE REDES NOMINALES LOCALES",
        "=" * 72,
        f"Fecha: {datetime.now().astimezone().isoformat()}",
        f"Permutaciones Freedman–Lane: {permutations:,}; semilla: {seed}.",
        "",
        "DISEÑO",
        f"- Descubrimiento local congelado: {len(local_selected)} filas p ingenua<.05.",
        f"- Blancos únicos canal×banda: {len(targets)}.",
        f"- Firmas de red signadas: {len(networks)}.",
        f"- Moderadores LEMON utilizables: {len(definitions)}.",
        f"- Modelos primarios firma×W: {len(network_results)}.",
        f"- Modelos secundarios canal×banda×W: {len(feature_results)}.",
        "- BAI local se transporta conceptualmente a STAI rasgo.",
        "- MSPSS local se transporta a MSPSS_total.",
        "- OC, OA y delta OC−OA permanecen separados; unidad=sujeto, n máximo=135.",
        "",
        "INFERENCIA",
        "- Modelo: Y ~ X + W + X:W, con Y/X estandarizados.",
        "- W continuas transformadas según diccionario y estandarizadas; binarias 0/1.",
        "- p ingenua clásica, HC3, Freedman–Lane, BH/BY y maxT.",
        "- La selección local es independiente; la búsqueda sobre 14 W externas es exploratoria.",
        "",
        "RESUMEN PRIMARIO: REDES",
        f"- p ingenua<.05: {int(network_results['naive_p_lt_0_05'].sum())}.",
        f"- p ingenua y permutación<.05: {int(network_results['robust_nominal'].sum())}.",
        f"- p ingenua, HC3 y permutación<.05: {int(network_results['triangulated_nominal'].sum())}.",
        f"- q BH perm global<.10: {int((network_results['q_BH_permutation_global'] < .10).sum())}.",
        f"- maxT global<.05: {int((network_results['p_maxT_global'] < .05).sum())}.",
        "",
        "RESUMEN SECUNDARIO: CARACTERÍSTICAS",
        f"- p ingenua<.05: {int(feature_results['naive_p_lt_0_05'].sum())}.",
        f"- p ingenua y permutación<.05: {int(feature_results['robust_nominal'].sum())}.",
        f"- p ingenua, HC3 y permutación<.05: {int(feature_results['triangulated_nominal'].sum())}.",
        f"- q BH perm global<.10: {int((feature_results['q_BH_permutation_global'] < .10).sum())}.",
        f"- maxT global<.05: {int((feature_results['p_maxT_global'] < .05).sum())}.",
        "",
        "EFECTOS PRINCIPALES EXTERNOS (PREGUNTA DISTINTA A MODERACIÓN)",
        f"- p ingenua<.05: {int((network_main['p_raw'] < .05).sum())}.",
        f"- q BH perm<.10: {int((network_main['q_BH_permutation'] < .10).sum())}.",
        f"- maxT<.05: {int((network_main['p_maxT'] < .05).sum())}.",
        f"- Características q BH perm<.10: {int((feature_main['q_BH_permutation'] < .10).sum())}.",
        f"- Características maxT<.05: {int((feature_main['p_maxT'] < .05).sum())}.",
        "- Estos resultados apoyan covariación EEG–outcome externa; no replican X×W.",
        "",
        "EXTENSIÓN CONCEPTUAL W4 → SIN PAREJA",
        f"- Modelos: {len(conceptual_w4)}.",
        f"- p perm<.05: {int((conceptual_w4['p_permutation'] < .05).sum()) if len(conceptual_w4) else 0}.",
        f"- q conceptual<.10: {int((conceptual_w4.get('q_BH_permutation_conceptual_W4', pd.Series(dtype=float)) < .10).sum()) if len(conceptual_w4) else 0}.",
        "- Sin pareja no equivale a vivir solo/a.",
        "",
        "TOP 15 REDES POR p DE PERMUTACIÓN",
    ]
    for row in top_network.itertuples(index=False):
        lines.append(
            f"- {row.network_id} × {row.external_moderator}: "
            f"β={row.beta_interaction_std:.5g}; p={row.p_raw:.5g}; "
            f"pHC3={row.p_hc3:.5g}; pperm={row.p_permutation:.5g}; "
            f"q={row.q_BH_permutation_global:.5g}; "
            f"pmaxT={row.p_maxT_global:.5g}."
        )
    lines.extend(["", "TOP 15 CARACTERÍSTICAS POR p DE PERMUTACIÓN"])
    for row in top_feature.itertuples(index=False):
        lines.append(
            f"- {row.target_id} × {row.external_moderator}: "
            f"β={row.beta_interaction_std:.5g}; p={row.p_raw:.5g}; "
            f"pHC3={row.p_hc3:.5g}; pperm={row.p_permutation:.5g}; "
            f"q={row.q_BH_permutation_global:.5g}; "
            f"pmaxT={row.p_maxT_global:.5g}."
        )
    lines.extend(["", "EFECTOS PRINCIPALES DE LAS 8 REDES"])
    for row in network_main.sort_values("p_permutation").itertuples(index=False):
        lines.append(
            f"- {row.network_id}: r={row.r_pearson:.5g}; "
            f"p={row.p_raw:.5g}; pperm={row.p_permutation:.5g}; "
            f"q={row.q_BH_permutation:.5g}; pmaxT={row.p_maxT:.5g}."
        )
    lines.extend(["", "TOP 10 EFECTOS PRINCIPALES CANAL×BANDA"])
    for row in feature_main.sort_values("p_permutation").head(10).itertuples(index=False):
        lines.append(
            f"- {row.target_id}: r={row.r_pearson:.5g}; "
            f"p={row.p_raw:.5g}; pperm={row.p_permutation:.5g}; "
            f"q={row.q_BH_permutation:.5g}; pmaxT={row.p_maxT:.5g}."
        )
    lines.extend(
        [
            "",
            "INTERPRETACIÓN",
            "- Una interacción con otra W es extensión conceptual, no réplica exacta.",
            "- STAI rasgo no es BAI; cualquier concordancia es de constructo ansiedad.",
            "- Los resultados nominales sin corrección solo generan hipótesis.",
            "- No se probaron variables demográficas después de ver un hit: las 14 W",
            "  fueron inventariadas y congeladas antes de ajustar asociaciones.",
            "- No se leyó data/local/local_outcomes_locked.csv.",
        ]
    )
    (output / "informe_validacion_LEMON.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_file_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest_archivos.csv":
            rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(rows).to_csv(
        output / "manifest_archivos.csv", index=False
    )


def main() -> int:
    args = parse_args()
    if args.permutations < 100:
        raise ValueError("Use al menos 100 permutaciones.")
    if "OneDrive" in str(args.project) or "OneDrive" in str(args.output):
        raise ValueError("Este pipeline no admite rutas OneDrive.")
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    started_at = datetime.now().astimezone()
    logger.info("Inicio de validación externa LEMON.")

    local_selected, targets, networks, components, local_hashes = (
        load_local_discovery(args.project)
    )
    mapping = pd.read_csv(args.project / "output" / "eeg_variable_mapping.csv")
    lemon_data, representations, eeg_long, lemon_hashes = load_lemon_data(
        args.lemon_repo, mapping
    )
    moderator_values, moderator_definitions, moderator_quality = (
        build_lemon_moderators(lemon_data)
    )
    variable_inventory = build_variable_inventory(
        lemon_data, moderator_definitions
    )
    logger.info(
        "LEMON n=%d; W retenidas=%d; redes=%d; blancos=%d.",
        len(lemon_data),
        len(moderator_definitions),
        len(networks),
        len(targets),
    )

    local_selected.to_csv(
        args.output / "descubrimiento_local_51_congelado.csv", index=False
    )
    targets.to_csv(args.output / "blancos_unicos_50.csv", index=False)
    networks.to_csv(args.output / "redes_locales_8.csv", index=False)
    components.to_csv(
        args.output / "componentes_redes_51.csv", index=False
    )
    moderator_values.to_csv(
        args.output / "moderadores_LEMON_135.csv", index=False
    )
    moderator_definitions.to_csv(
        args.output / "diccionario_moderadores_LEMON_14.csv", index=False
    )
    moderator_quality.to_csv(
        args.output / "control_calidad_moderadores.csv", index=False
    )
    variable_inventory.to_csv(
        args.output / "inventario_todas_variables_LEMON.csv", index=False
    )

    network_scores = construct_network_scores(
        representations, networks, components
    )
    network_scores.insert(0, "ID", lemon_data["ID"].to_numpy())
    network_scores.to_csv(
        args.output / "firmas_red_LEMON_135.csv", index=False
    )

    feature_scores = pd.DataFrame({"ID": lemon_data["ID"]})
    for target in targets.itertuples(index=False):
        feature_scores[target.target_id] = representations[
            target.representation
        ][target.eeg_id].to_numpy(float)
    feature_scores.to_csv(
        args.output / "blancos_canal_banda_LEMON_135.csv", index=False
    )

    rng = np.random.default_rng(args.seed)
    random_keys = rng.random(
        (args.permutations, len(lemon_data)), dtype=np.float32
    )
    global_orders = np.argsort(
        random_keys, axis=1, kind="stable"
    ).astype(np.int16)
    np.savez_compressed(
        args.output / "permutation_scheme.npz",
        random_keys=random_keys,
        global_orders=global_orders,
        seed=np.array(args.seed, dtype=np.int64),
    )

    merged_moderators = lemon_data[["ID"]].merge(
        moderator_values, on="ID", how="left"
    )
    network_results, null_network = run_interaction_models(
        "network",
        moderator_definitions,
        merged_moderators,
        lemon_data,
        network_scores,
        networks,
        "network_id",
        random_keys,
        logger,
        args.output / "progress.json",
    )
    feature_results, null_feature = run_interaction_models(
        "feature",
        moderator_definitions,
        merged_moderators,
        lemon_data,
        feature_scores,
        targets,
        "target_id",
        random_keys,
        logger,
        args.output / "progress.json",
    )
    network_main, null_network_main = main_effects(
        "network",
        lemon_data,
        network_scores,
        networks,
        "network_id",
        global_orders,
    )
    feature_main, null_feature_main = main_effects(
        "feature",
        lemon_data,
        feature_scores,
        targets,
        "target_id",
        global_orders,
    )
    conceptual_w4 = conceptual_w4_subset(network_results)

    network_results.to_csv(
        args.output / "resultados_112_redes_por_W.csv", index=False
    )
    feature_results.to_csv(
        args.output / "resultados_700_caracteristicas_por_W.csv", index=False
    )
    network_main.to_csv(
        args.output / "efectos_principales_8_redes.csv", index=False
    )
    feature_main.to_csv(
        args.output / "efectos_principales_50_caracteristicas.csv", index=False
    )
    conceptual_w4.to_csv(
        args.output / "extension_conceptual_W4_sin_pareja.csv", index=False
    )
    network_results.loc[network_results["naive_p_lt_0_05"]].to_csv(
        args.output / "p_ingenuas_redes_lt_0_05.csv", index=False
    )
    feature_results.loc[feature_results["naive_p_lt_0_05"]].to_csv(
        args.output / "p_ingenuas_caracteristicas_lt_0_05.csv", index=False
    )
    network_results.loc[network_results["robust_nominal"]].to_csv(
        args.output / "coincidencias_nominales_redes.csv", index=False
    )
    feature_results.loc[feature_results["robust_nominal"]].to_csv(
        args.output / "coincidencias_nominales_caracteristicas.csv",
        index=False,
    )
    network_results.loc[network_results["triangulated_nominal"]].to_csv(
        args.output / "coincidencias_trianguladas_redes.csv", index=False
    )
    feature_results.loc[feature_results["triangulated_nominal"]].to_csv(
        args.output / "coincidencias_trianguladas_caracteristicas.csv",
        index=False,
    )
    np.savez_compressed(
        args.output / "null_arrays_interactions.npz",
        network_abs_t=null_network,
        feature_abs_t=null_feature,
    )
    np.savez_compressed(
        args.output / "null_arrays_main_effects.npz",
        network_abs_r=null_network_main,
        feature_abs_r=null_feature_main,
    )
    plot_network_heatmap(
        network_results, networks, moderator_definitions, args.output
    )
    plot_main_effects(network_main, feature_main, args.output)
    write_report(
        args.output,
        local_selected,
        targets,
        networks,
        moderator_definitions,
        network_results,
        feature_results,
        network_main,
        feature_main,
        conceptual_w4,
        args.permutations,
        args.seed,
    )

    source_hashes = {**local_hashes, **lemon_hashes}
    source_hashes_after = {
        path: sha256_file(Path(path)) for path in source_hashes
    }
    if source_hashes != source_hashes_after:
        raise RuntimeError("Cambió al menos un archivo fuente durante la corrida.")
    elapsed = time.perf_counter() - started
    manifest = {
        "analysis": "LEMON_nominal_local_network_validation_all_W",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "local_nominal_rows": len(local_selected),
        "unique_feature_targets": len(targets),
        "local_networks": len(networks),
        "lemon_subjects": len(lemon_data),
        "lemon_moderators": len(moderator_definitions),
        "network_interaction_models": len(network_results),
        "feature_interaction_models": len(feature_results),
        "permutations": args.permutations,
        "seed": args.seed,
        "network_naive_hits": int(network_results["naive_p_lt_0_05"].sum()),
        "network_robust_nominal_hits": int(network_results["robust_nominal"].sum()),
        "network_triangulated_nominal_hits": int(
            network_results["triangulated_nominal"].sum()
        ),
        "network_q_perm_global_lt_0_10": int(
            (network_results["q_BH_permutation_global"] < 0.10).sum()
        ),
        "network_maxT_global_lt_0_05": int(
            (network_results["p_maxT_global"] < 0.05).sum()
        ),
        "feature_naive_hits": int(feature_results["naive_p_lt_0_05"].sum()),
        "feature_robust_nominal_hits": int(feature_results["robust_nominal"].sum()),
        "feature_triangulated_nominal_hits": int(
            feature_results["triangulated_nominal"].sum()
        ),
        "feature_q_perm_global_lt_0_10": int(
            (feature_results["q_BH_permutation_global"] < 0.10).sum()
        ),
        "feature_maxT_global_lt_0_05": int(
            (feature_results["p_maxT_global"] < 0.05).sum()
        ),
        "network_main_q_perm_lt_0_10": int(
            (network_main["q_BH_permutation"] < 0.10).sum()
        ),
        "network_main_maxT_lt_0_05": int(
            (network_main["p_maxT"] < 0.05).sum()
        ),
        "feature_main_q_perm_lt_0_10": int(
            (feature_main["q_BH_permutation"] < 0.10).sum()
        ),
        "feature_main_maxT_lt_0_05": int(
            (feature_main["p_maxT"] < 0.05).sum()
        ),
        "source_hashes_before": source_hashes,
        "source_hashes_after": source_hashes_after,
        "sources_preserved": True,
        "one_drive_used": False,
        "data_local_locked_read": False,
        "versions": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    atomic_json(args.output / "analysis_manifest.json", manifest)
    atomic_json(
        args.output / "progress.json",
        {
            "status": "completed",
            "stage": "finished",
            "completed": args.permutations,
            "total": args.permutations,
            "percent": 100.0,
            "elapsed_seconds": elapsed,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )
    logger.info(
        "Fin %.2f s | redes q<.10=%d maxT<.05=%d | "
        "features q<.10=%d maxT<.05=%d.",
        elapsed,
        manifest["network_q_perm_global_lt_0_10"],
        manifest["network_maxT_global_lt_0_05"],
        manifest["feature_q_perm_global_lt_0_10"],
        manifest["feature_maxT_global_lt_0_05"],
    )
    logging.shutdown()
    write_file_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
