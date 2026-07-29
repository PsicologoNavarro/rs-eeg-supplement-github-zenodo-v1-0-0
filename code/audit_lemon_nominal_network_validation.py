from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


N = 135
B = 10_000
EEG_IDS = [f"EEG{i}" for i in range(1, 65)]
OUTCOME_MAP = {"BAI": "STAI_Trait_Anxiety", "MSPSS": "MSPSS_total"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoría independiente LEMON.")
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
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path.cwd() / "output_lemon_nominal_network_validation",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path.cwd() / "output_lemon_nominal_network_validation_audit",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / values.std(ddof=1)


def bh(p_values: np.ndarray, by: bool = False) -> np.ndarray:
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    factor = np.sum(1 / np.arange(1, len(ranked) + 1)) if by else 1.0
    adjusted = ranked * len(ranked) * factor / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def numeric_text(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:[.,]\d+)?", str(value))
    return float(match.group().replace(",", ".")) if match else np.nan


def midpoint(value: object) -> float:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    return (float(numbers[0]) + float(numbers[1])) / 2 if len(numbers) >= 2 else np.nan


def close_check(
    name: str, observed: np.ndarray, expected: np.ndarray, tolerance: float
) -> dict:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    same_shape = observed.shape == expected.shape
    difference = (
        float(np.nanmax(np.abs(observed - expected)))
        if same_shape and observed.size
        else np.inf
    )
    nan_same = same_shape and np.array_equal(np.isnan(observed), np.isnan(expected))
    return {
        "check": name,
        "passed": bool(same_shape and nan_same and difference <= tolerance),
        "max_abs_difference": difference,
        "tolerance": tolerance,
    }


def rebuild_moderators(ids: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame({"ID": ids}).merge(meta, on="ID", how="left")
    output = pd.DataFrame({"ID": ids})
    gender = pd.to_numeric(data["Gender_ 1=female_2=male"], errors="coerce")
    output["GENDER_FEMALE"] = gender.map({1.0: 1.0, 2.0: 0.0})
    output["AGE_MIDPOINT"] = data["Age"].map(midpoint)
    handed = data["Handedness"].astype("string").str.strip().str.lower()
    output["HANDEDNESS_NONRIGHT"] = np.where(
        handed.eq("right").fillna(False),
        0.0,
        np.where(handed.notna(), 1.0, np.nan),
    )
    education = data["Education"].astype("string").str.strip().str.lower()
    high = education.isin(["gymnasium", "gymansium"])
    valid = high | education.str.contains(
        "realschule|hauptschule|none", regex=True, na=False
    )
    education_lower = pd.Series(np.nan, index=data.index)
    education_lower.loc[valid] = (~high.loc[valid]).astype(float)
    output["EDUCATION_LOWER"] = education_lower
    output["DRUG_SCREEN_POSITIVE"] = pd.to_numeric(
        data["DRUG_0=negative_1=Positive"], errors="coerce"
    )
    smoking = pd.to_numeric(
        data["Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)"],
        errors="coerce",
    )
    output["SMOKING_ANY"] = np.where(
        smoking.notna(), (smoking > 1).astype(float), np.nan
    )
    skid = data["SKID_Diagnoses"].astype("string").str.strip().str.lower()
    output["ANY_SKID_DIAGNOSIS"] = np.where(
        skid.isna(),
        np.nan,
        np.where(skid.str.fullmatch(r"none\s*", na=False), 0.0, 1.0),
    )
    output["HAMILTON_LOG1P"] = np.log1p(
        pd.to_numeric(data["Hamilton_Scale"], errors="coerce")
    )
    output["BSL23_SUM_LOG1P"] = np.log1p(
        pd.to_numeric(data["BSL23_sumscore"], errors="coerce")
    )
    behavior = pd.to_numeric(data["BSL23_behavior"], errors="coerce")
    output["BSL23_BEHAVIOR_ANY"] = np.where(
        behavior.notna(), (behavior > 0).astype(float), np.nan
    )
    output["AUDIT_LOG1P"] = np.log1p(
        pd.to_numeric(data["AUDIT"], errors="coerce")
    )
    output["ALCOHOL_UNITS_LOG1P"] = np.log1p(
        data["Standard_Alcoholunits_Last_28days"].map(numeric_text)
    )
    family = (
        data["Alcohol_Dependence_In_1st-3rd_Degree_relative"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    output["FAMILY_ALCOHOL_DEPENDENCE"] = np.where(
        family.eq("no").fillna(False),
        0.0,
        np.where(family.str.startswith("yes", na=False), 1.0, np.nan),
    )
    relationship = (
        data["Relationship_Status"].astype("string").str.strip().str.lower()
    )
    output["RELATIONSHIP_NO_PARTNER"] = np.where(
        relationship.eq("yes").fillna(False),
        0.0,
        np.where(relationship.eq("no").fillna(False), 1.0, np.nan),
    )
    return output


def ols_stats(
    y_raw: np.ndarray, x_raw: np.ndarray, w_raw: np.ndarray, w_type: str
) -> tuple[dict, dict]:
    mask = np.isfinite(y_raw) & np.isfinite(x_raw) & np.isfinite(w_raw)
    y = z(y_raw[mask])
    x = z(x_raw[mask])
    original_w = w_raw[mask]
    w = original_w if w_type == "binary" else z(original_w)
    w = w - w.mean()
    interaction = x * w
    reduced = np.column_stack([np.ones(len(y)), x, w])
    full = np.column_stack([reduced, interaction])
    inverse = np.linalg.inv(full.T @ full)
    beta = inverse @ full.T @ y
    residual = y - full @ beta
    df = len(y) - 4
    mse = float(residual @ residual) / df
    se = np.sqrt(np.diag(inverse) * mse)
    t_value = beta[3] / se[3]
    p_value = 2 * stats.t.sf(abs(t_value), df)
    leverage = np.sum((full @ inverse) * full, axis=1)
    adjusted = residual / (1 - leverage)
    meat = full.T @ ((adjusted[:, None] ** 2) * full)
    hc3 = inverse @ meat @ inverse
    se_hc3 = math.sqrt(hc3[3, 3])
    t_hc3 = beta[3] / se_hc3
    p_hc3 = 2 * stats.t.sf(abs(t_hc3), df)
    inverse_reduced = np.linalg.inv(reduced.T @ reduced)
    reduced_residual = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    z_residual = interaction - reduced @ (
        inverse_reduced @ reduced.T @ interaction
    )
    stats_out = {
        "n": len(y),
        "beta": beta[3],
        "se": se[3],
        "t": t_value,
        "p": p_value,
        "se_hc3": se_hc3,
        "t_hc3": t_hc3,
        "p_hc3": p_hc3,
    }
    auxiliary = {
        "mask": mask,
        "reduced": reduced,
        "inverse_reduced": inverse_reduced,
        "reduced_residual": reduced_residual,
        "z_residual": z_residual,
        "z_norm2": float(z_residual @ z_residual),
        "df": df,
    }
    return stats_out, auxiliary


def permutation_t(auxiliary: dict, orders: np.ndarray) -> np.ndarray:
    residual_permuted = auxiliary["reduced_residual"][orders]
    projection = residual_permuted @ auxiliary["reduced"]
    q_norm2 = np.sum(residual_permuted**2, axis=1) - np.sum(
        (projection @ auxiliary["inverse_reduced"]) * projection, axis=1
    )
    numerator = residual_permuted @ auxiliary["z_residual"]
    partial_r = numerator / np.sqrt(q_norm2 * auxiliary["z_norm2"])
    partial_r = np.clip(partial_r, -1 + 1e-12, 1 - 1e-12)
    return partial_r * np.sqrt(auxiliary["df"] / (1 - partial_r**2))


def main() -> int:
    cfg = parse_args()
    if any("OneDrive" in str(path) for path in [cfg.project, cfg.analysis_output, cfg.audit_output]):
        raise ValueError("La auditoría no admite OneDrive.")
    cfg.audit_output.mkdir(parents=True, exist_ok=True)
    out = cfg.analysis_output
    checks: list[dict] = []

    manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    local = pd.read_csv(out / "descubrimiento_local_51_congelado.csv")
    targets = pd.read_csv(out / "blancos_unicos_50.csv")
    networks = pd.read_csv(out / "redes_locales_8.csv")
    components = pd.read_csv(out / "componentes_redes_51.csv")
    moderators_saved = pd.read_csv(out / "moderadores_LEMON_135.csv")
    definitions = pd.read_csv(out / "diccionario_moderadores_LEMON_14.csv")
    network_scores_saved = pd.read_csv(out / "firmas_red_LEMON_135.csv")
    feature_scores_saved = pd.read_csv(out / "blancos_canal_banda_LEMON_135.csv")
    network_result = pd.read_csv(out / "resultados_112_redes_por_W.csv")
    feature_result = pd.read_csv(out / "resultados_700_caracteristicas_por_W.csv")
    network_main = pd.read_csv(out / "efectos_principales_8_redes.csv")
    feature_main = pd.read_csv(out / "efectos_principales_50_caracteristicas.csv")
    scheme = np.load(out / "permutation_scheme.npz")
    null_interaction = np.load(out / "null_arrays_interactions.npz")
    null_main = np.load(out / "null_arrays_main_effects.npz")

    checks.extend(
        [
            {
                "check": "estructura_local_51_50_8",
                "passed": bool(
                    len(local) == 51
                    and len(targets) == 50
                    and len(networks) == 8
                    and len(components) == 51
                ),
            },
            {
                "check": "conteos_locales_OC_OA_delta",
                "passed": local.groupby("analysis_family").size().to_dict()
                == {"DELTA_OC_MINUS_OA": 18, "OA": 2, "OC": 31},
            },
            {
                "check": "pesos_red_suman_uno_absoluto",
                "passed": bool(
                    np.allclose(
                        components.groupby("network_id")["weight_sign_equal"]
                        .apply(lambda x: np.abs(x).sum())
                        .to_numpy(),
                        1,
                    )
                ),
            },
            {
                "check": "signos_pesos_coinciden_beta_local",
                "passed": bool(
                    np.array_equal(
                        np.sign(components["weight_sign_equal"]),
                        np.sign(components["local_beta_interaction"]),
                    )
                ),
            },
            {
                "check": "estructura_LEMON_135_y_14W",
                "passed": bool(
                    len(moderators_saved) == N
                    and moderators_saved["ID"].nunique() == N
                    and len(definitions) == 14
                ),
            },
            {
                "check": "estructura_modelos_112_y_700",
                "passed": bool(
                    len(network_result) == 112 and len(feature_result) == 700
                ),
            },
        ]
    )

    meta_path = (
        cfg.lemon_repo
        / "data"
        / "lemon"
        / "metadata"
        / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    )
    meta = pd.read_csv(meta_path)
    moderators_rebuilt = rebuild_moderators(moderators_saved["ID"], meta)
    checks.append(
        close_check(
            "reconstruccion_14_moderadores_desde_fuente",
            moderators_saved.drop(columns="ID").to_numpy(float),
            moderators_rebuilt.drop(columns="ID").to_numpy(float),
            2e-12,
        )
    )

    eeg_path = (
        cfg.lemon_repo
        / "outputs"
        / "06_features"
        / "participant_condition_bandpower.csv"
    )
    mapping = pd.read_csv(cfg.project / "output" / "eeg_variable_mapping.csv")
    eeg = pd.read_csv(eeg_path)
    eeg = eeg.loc[eeg["participant_id"].isin(moderators_saved["ID"])].copy()
    checks.append(
        {
            "check": "EEG_17280_sin_faltantes_duplicados",
            "passed": bool(
                len(eeg) == 17_280
                and not eeg["participant_median_log_power"].isna().any()
                and not eeg.duplicated(
                    ["participant_id", "condition", "channel", "band"]
                ).any()
            ),
        }
    )
    pivot = eeg.pivot(
        index="participant_id",
        columns=["condition", "channel", "band"],
        values="participant_median_log_power",
    ).reindex(moderators_saved["ID"])
    representations = {}
    for name, condition in {"OC": "eyes_closed", "OA": "eyes_open"}.items():
        matrix = pd.DataFrame(index=moderators_saved["ID"], columns=EEG_IDS, dtype=float)
        for row in mapping.itertuples(index=False):
            matrix[row.eeg_id] = pivot[
                (condition, row.channel, row.band)
            ].to_numpy(float)
        representations[name] = matrix
    representations["DELTA_OC_MINUS_OA"] = representations["OC"] - representations["OA"]

    network_rebuilt = pd.DataFrame({"ID": moderators_saved["ID"]})
    for network in networks.itertuples(index=False):
        block = components.loc[components["network_id"].eq(network.network_id)]
        raw = representations[network.representation][block["eeg_id"]].to_numpy(float)
        standardized = np.column_stack([z(raw[:, index]) for index in range(raw.shape[1])])
        network_rebuilt[network.network_id] = (
            standardized @ block["weight_sign_equal"].to_numpy(float)
        )
    checks.append(
        close_check(
            "reconstruccion_8_firmas_desde_EEG",
            network_scores_saved.drop(columns="ID").to_numpy(float),
            network_rebuilt.drop(columns="ID").to_numpy(float),
            2e-12,
        )
    )
    feature_rebuilt = pd.DataFrame({"ID": moderators_saved["ID"]})
    for target in targets.itertuples(index=False):
        feature_rebuilt[target.target_id] = representations[
            target.representation
        ][target.eeg_id].to_numpy(float)
    checks.append(
        close_check(
            "reconstruccion_50_blancos_desde_EEG",
            feature_scores_saved.drop(columns="ID").to_numpy(float),
            feature_rebuilt.drop(columns="ID").to_numpy(float),
            2e-12,
        )
    )

    keys = scheme["random_keys"]
    orders_global = scheme["global_orders"].astype(int)
    checks.append(
        {
            "check": "esquema_10000_permutaciones_valido",
            "passed": bool(
                keys.shape == (B, N)
                and orders_global.shape == (B, N)
                and np.all(np.sort(orders_global, axis=1) == np.arange(N))
                and int(scheme["seed"]) == 20260727
                and np.array_equal(
                    orders_global,
                    np.argsort(keys, axis=1, kind="stable"),
                )
            ),
        }
    )

    eligible_path = (
        cfg.lemon_repo / "outputs" / "02_cohort" / "lemon_eligible_subjects.csv"
    )
    outcomes = pd.DataFrame({"ID": moderators_saved["ID"]}).merge(
        pd.read_csv(eligible_path), on="ID", how="left"
    )
    moderator_kind = definitions.set_index("moderator")["type"].to_dict()
    permutation_cache = {}
    for moderator in definitions["moderator"]:
        mask = moderators_saved[moderator].notna().to_numpy()
        permutation_cache[moderator] = np.argsort(
            keys[:, mask], axis=1, kind="stable"
        ).astype(np.int16)

    def audit_interactions(
        result: pd.DataFrame,
        score_table: pd.DataFrame,
        id_column: str,
        null_saved: np.ndarray,
        prefix: str,
    ) -> None:
        recalculated_rows = []
        null_columns = []
        for row in result.itertuples(index=False):
            target_id = getattr(row, id_column)
            y = outcomes[row.lemon_outcome].to_numpy(float)
            x = score_table[target_id].to_numpy(float)
            w = moderators_saved[row.external_moderator].to_numpy(float)
            estimates, auxiliary = ols_stats(
                y, x, w, moderator_kind[row.external_moderator]
            )
            recalculated_rows.append(estimates)
            null_columns.append(
                np.abs(
                    permutation_t(
                        auxiliary, permutation_cache[row.external_moderator]
                    )
                ).astype(np.float32)
            )
        recalculated = pd.DataFrame(recalculated_rows)
        null_recalculated = np.column_stack(null_columns)
        checks.extend(
            [
                close_check(
                    f"{prefix}_beta_OLS",
                    result["beta_interaction_std"],
                    recalculated["beta"],
                    2e-12,
                ),
                close_check(
                    f"{prefix}_t_y_p_clasicos",
                    result[["t_interaction", "p_raw"]].to_numpy(float),
                    recalculated[["t", "p"]].to_numpy(float),
                    3e-12,
                ),
                close_check(
                    f"{prefix}_HC3",
                    result[["se_hc3", "t_hc3", "p_hc3"]].to_numpy(float),
                    recalculated[["se_hc3", "t_hc3", "p_hc3"]].to_numpy(float),
                    3e-11,
                ),
                close_check(
                    f"{prefix}_nulos_Freedman_Lane",
                    null_saved,
                    null_recalculated,
                    2e-6,
                ),
            ]
        )
        observed = result["t_interaction"].abs().to_numpy(float)
        exceed = np.sum(null_recalculated >= observed[None, :] - 1e-12, axis=0)
        p_perm = (exceed + 1) / (B + 1)
        global_max = null_recalculated.max(axis=1)
        p_max = np.array(
            [
                (np.count_nonzero(global_max >= value - 1e-12) + 1) / (B + 1)
                for value in observed
            ]
        )
        checks.extend(
            [
                close_check(
                    f"{prefix}_p_permutacion",
                    result["p_permutation"],
                    p_perm,
                    2e-15,
                ),
                close_check(
                    f"{prefix}_BH_permutacion_global",
                    result["q_BH_permutation_global"],
                    bh(p_perm),
                    2e-15,
                ),
                close_check(
                    f"{prefix}_BY_permutacion_global",
                    result["q_BY_permutation_global"],
                    bh(p_perm, by=True),
                    2e-15,
                ),
                close_check(
                    f"{prefix}_maxT_global",
                    result["p_maxT_global"],
                    p_max,
                    2e-15,
                ),
                {
                    "check": f"{prefix}_banderas_nominales",
                    "passed": bool(
                        np.array_equal(
                            result["triangulated_nominal"].to_numpy(bool),
                            (
                                (result["p_raw"] < 0.05)
                                & (result["p_hc3"] < 0.05)
                                & (result["p_permutation"] < 0.05)
                            ).to_numpy(bool),
                        )
                    ),
                },
            ]
        )

    audit_interactions(
        network_result,
        network_scores_saved,
        "network_id",
        null_interaction["network_abs_t"],
        "redes112",
    )
    audit_interactions(
        feature_result,
        feature_scores_saved,
        "target_id",
        null_interaction["feature_abs_t"],
        "features700",
    )

    def audit_main(
        result: pd.DataFrame,
        score_table: pd.DataFrame,
        id_column: str,
        null_saved: np.ndarray,
        prefix: str,
    ) -> None:
        observed = []
        p_raw = []
        null_columns = []
        for row in result.itertuples(index=False):
            target_id = getattr(row, id_column)
            x = z(score_table[target_id].to_numpy(float))
            y = z(outcomes[row.lemon_outcome].to_numpy(float))
            observed.append(float(x @ y / (N - 1)))
            p_raw.append(float(stats.pearsonr(x, y).pvalue))
            null_columns.append(
                np.abs(y[orders_global] @ x / (N - 1)).astype(np.float32)
            )
        observed = np.asarray(observed)
        null_recalculated = np.column_stack(null_columns)
        exceed = np.sum(
            null_recalculated >= np.abs(observed)[None, :] - 1e-12, axis=0
        )
        p_perm = (exceed + 1) / (B + 1)
        max_null = null_recalculated.max(axis=1)
        p_max = np.array(
            [
                (np.count_nonzero(max_null >= value - 1e-12) + 1) / (B + 1)
                for value in np.abs(observed)
            ]
        )
        checks.extend(
            [
                close_check(
                    f"{prefix}_r_y_p",
                    result[["r_pearson", "p_raw"]].to_numpy(float),
                    np.column_stack([observed, p_raw]),
                    2e-12,
                ),
                close_check(
                    f"{prefix}_nulos",
                    null_saved,
                    null_recalculated,
                    2e-7,
                ),
                close_check(
                    f"{prefix}_p_perm_q_y_maxT",
                    result[
                        ["p_permutation", "q_BH_permutation", "p_maxT"]
                    ].to_numpy(float),
                    np.column_stack([p_perm, bh(p_perm), p_max]),
                    2e-15,
                ),
            ]
        )

    audit_main(
        network_main,
        network_scores_saved,
        "network_id",
        null_main["network_abs_r"],
        "main_redes8",
    )
    audit_main(
        feature_main,
        feature_scores_saved,
        "target_id",
        null_main["feature_abs_r"],
        "main_features50",
    )

    file_manifest = pd.read_csv(out / "manifest_archivos.csv")
    bad_output_hashes = []
    for row in file_manifest.itertuples(index=False):
        path = out / row.relative_path
        if not path.is_file() or sha256(path) != row.sha256:
            bad_output_hashes.append(row.relative_path)
    checks.append(
        {
            "check": "manifest_salida_SHA256",
            "passed": not bad_output_hashes,
            "bad_files": bad_output_hashes,
        }
    )
    bad_sources = []
    for path_string, expected in manifest["source_hashes_before"].items():
        path = Path(path_string)
        if not path.is_file() or sha256(path) != expected:
            bad_sources.append(path_string)
    checks.extend(
        [
            {
                "check": "fuentes_preservadas_SHA256",
                "passed": not bad_sources,
                "bad_files": bad_sources,
            },
            {
                "check": "sin_OneDrive_y_sin_local_locked",
                "passed": bool(
                    not manifest["one_drive_used"]
                    and not manifest["data_local_locked_read"]
                    and all(
                        "OneDrive" not in path
                        and "data\\local\\local_outcomes_locked.csv" not in path
                        for path in manifest["source_hashes_before"]
                    )
                ),
            },
            {
                "check": "cero_interacciones_FDR_o_maxT",
                "passed": bool(
                    (network_result["q_BH_permutation_global"] < 0.10).sum() == 0
                    and (feature_result["q_BH_permutation_global"] < 0.10).sum() == 0
                    and (network_result["p_maxT_global"] < 0.05).sum() == 0
                    and (feature_result["p_maxT_global"] < 0.05).sum() == 0
                ),
            },
            {
                "check": "un_efecto_principal_red_maxT",
                "passed": bool(
                    (network_main["p_maxT"] < 0.05).sum() == 1
                    and network_main.loc[
                        network_main["p_maxT"].idxmin(), "network_id"
                    ]
                    == "NET03__DELTA_OC_MINUS_OA__MSPSS__W3"
                ),
            },
        ]
    )

    audit = pd.DataFrame(checks)
    audit.to_csv(cfg.audit_output / "auditoria_checks.csv", index=False)
    failures = audit.loc[~audit["passed"]]
    summary = {
        "audit_passed": failures.empty,
        "checks_total": len(audit),
        "checks_failed": len(failures),
        "local_nominal_rows": len(local),
        "lemon_n": N,
        "moderators": len(definitions),
        "network_models": len(network_result),
        "feature_models": len(feature_result),
        "network_triangulated_nominal": int(
            network_result["triangulated_nominal"].sum()
        ),
        "feature_triangulated_nominal": int(
            feature_result["triangulated_nominal"].sum()
        ),
        "network_interaction_FDR_hits": int(
            (network_result["q_BH_permutation_global"] < 0.10).sum()
        ),
        "network_interaction_maxT_hits": int(
            (network_result["p_maxT_global"] < 0.05).sum()
        ),
        "network_main_FDR_hits": int(
            (network_main["q_BH_permutation"] < 0.10).sum()
        ),
        "network_main_maxT_hits": int((network_main["p_maxT"] < 0.05).sum()),
    }
    (cfg.audit_output / "auditoria_resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    network_result.loc[network_result["triangulated_nominal"]].sort_values(
        "p_permutation"
    ).to_csv(
        cfg.audit_output / "redes_nominales_trianguladas_auditadas.csv",
        index=False,
    )
    network_main.sort_values("p_permutation").to_csv(
        cfg.audit_output / "efectos_principales_red_auditados.csv",
        index=False,
    )
    lines = [
        "AUDITORÍA INDEPENDIENTE — VALIDACIÓN LEMON",
        "=" * 60,
        f"Estado: {'APROBADA' if failures.empty else 'FALLIDA'}",
        f"Checks: {len(audit) - len(failures)}/{len(audit)} aprobados.",
        f"LEMON: n={N}; W={len(definitions)}.",
        f"Interacciones: {len(network_result)} redes + {len(feature_result)} características.",
        f"Coincidencias trianguladas de red: {summary['network_triangulated_nominal']}.",
        f"Interacciones de red FDR<.10: {summary['network_interaction_FDR_hits']}.",
        f"Interacciones de red maxT<.05: {summary['network_interaction_maxT_hits']}.",
        f"Efectos principales de red FDR<.10: {summary['network_main_FDR_hits']}.",
        f"Efectos principales de red maxT<.05: {summary['network_main_maxT_hits']}.",
        "",
        "Se reconstruyeron desde fuente las 14 W, las 8 firmas, los 50 blancos,",
        "los 812 modelos y todas las distribuciones nulas de 10,000 permutaciones.",
    ]
    if not failures.empty:
        lines.extend(["", "FALLAS", *[f"- {x}" for x in failures["check"]]])
    (cfg.audit_output / "informe_auditoria_LEMON.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0 if failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
