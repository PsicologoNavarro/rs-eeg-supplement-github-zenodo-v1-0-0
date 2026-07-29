from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


FAMILIES = ["OC", "OA", "DELTA_OC_MINUS_OA"]
FAMILY_LABELS = {
    "OC": "Ojos cerrados",
    "OA": "Ojos abiertos",
    "DELTA_OC_MINUS_OA": "Reactividad OC−OA",
}
OUTCOMES = ["BAI", "MSPSS"]
MODERATORS = [f"W{i}" for i in range(1, 6)]
EEG_IDS = [f"EEG{i}" for i in range(1, 65)]
SCREEN_ALPHA = 0.05
DF_RESID = 21


@dataclass
class ScreenPreparation:
    family: str
    outcome: str
    moderator: str
    combo_index: int
    x_raw: np.ndarray
    y: np.ndarray
    w: np.ndarray
    reduced_fitted: np.ndarray
    reduced_residuals: np.ndarray
    full_design: np.ndarray
    full_pinv: np.ndarray
    full_inv_interaction_diag: np.ndarray
    observed_beta: np.ndarray
    observed_t: np.ndarray
    observed_p: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Permutación selectiva completa de 1,920 moderaciones EEG."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--screen-alpha", type=float, default=SCREEN_ALPHA
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("selective_permutation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_inputs(project: Path) -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, dict[str, str]
]:
    paths = {
        "OC": project / "output" / "analysis_dataset_25.csv",
        "OA": (
            project
            / "output_oa_oc_delta_comparison"
            / "eyes_open"
            / "analysis_dataset_25.csv"
        ),
        "DELTA_OC_MINUS_OA": (
            project
            / "output_oa_oc_delta_comparison"
            / "delta_oc_minus_oa"
            / "analysis_dataset_25.csv"
        ),
    }
    comparison_path = (
        project
        / "output_oa_oc_delta_comparison"
        / "resultados_1920_comparados.csv"
    )
    mapping_path = project / "output" / "eeg_variable_mapping.csv"
    required_paths = [*paths.values(), comparison_path, mapping_path]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan insumos:\n" + "\n".join(missing))

    data = {
        family: pd.read_csv(path)
        .sort_values("subject_id")
        .reset_index(drop=True)
        for family, path in paths.items()
    }
    required = {
        "subject_id",
        "condition",
        *OUTCOMES,
        *MODERATORS,
        *EEG_IDS,
    }
    for family, frame in data.items():
        if frame.shape != (25, 73):
            raise ValueError(f"{family}: dimensión inesperada {frame.shape}.")
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"{family}: faltan {sorted(absent)}.")
        if frame["subject_id"].nunique() != 25:
            raise ValueError(f"{family}: no contiene 25 sujetos únicos.")
        numeric = frame[[*OUTCOMES, *MODERATORS, *EEG_IDS]].apply(
            pd.to_numeric, errors="raise"
        )
        if numeric.isna().any().any():
            raise ValueError(f"{family}: faltantes analíticos.")
        frame[[*OUTCOMES, *MODERATORS, *EEG_IDS]] = numeric

    invariant = ["subject_id", *OUTCOMES, *MODERATORS]
    if not data["OC"][invariant].equals(data["OA"][invariant]):
        raise ValueError("OC/OA no comparten sujetos, outcomes y moderadores.")
    if not data["OC"][invariant].equals(
        data["DELTA_OC_MINUS_OA"][invariant]
    ):
        raise ValueError("Delta no comparte sujetos, outcomes y moderadores.")
    expected_delta = (
        data["OC"][EEG_IDS].to_numpy(float)
        - data["OA"][EEG_IDS].to_numpy(float)
    )
    stored_delta = data["DELTA_OC_MINUS_OA"][EEG_IDS].to_numpy(float)
    if np.max(np.abs(expected_delta - stored_delta)) > 1e-12:
        raise ValueError("Delta no reconcilia exactamente con OC−OA.")

    comparison = pd.read_csv(comparison_path)
    if len(comparison) != 1920:
        raise ValueError(f"Comparación original: {len(comparison)} filas.")
    if comparison.groupby("analysis_family").size().to_dict() != {
        family: 640 for family in FAMILIES
    }:
        raise ValueError("La comparación no contiene 640 modelos por familia.")
    mapping = pd.read_csv(mapping_path)
    if len(mapping) != 64 or mapping["eeg_id"].tolist() != EEG_IDS:
        raise ValueError("El mapping no contiene EEG1..EEG64 en orden.")

    hashes = {str(path): sha256_file(path) for path in required_paths}
    for prior in [
        project / "output" / "resultados_640_modelos.csv",
        project
        / "output_spatial_aggregation"
        / "resultados_750_agregados.csv",
    ]:
        if prior.is_file():
            hashes[str(prior)] = sha256_file(prior)
    return data, comparison, mapping, hashes


def prepare_batched_design(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transpose = np.swapaxes(design, 1, 2)
    cross_product = np.einsum("fkn,fnj->fkj", transpose, design)
    inverse = np.linalg.inv(cross_product)
    pseudo_inverse = np.einsum("fkj,fjn->fkn", inverse, transpose)
    return inverse, pseudo_inverse


def fit_batched(
    y_matrix: np.ndarray,
    design: np.ndarray,
    pseudo_inverse: np.ndarray,
    inverse_interaction_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.einsum("fkn,fn->fk", pseudo_inverse, y_matrix)
    fitted = np.einsum("fnk,fk->fn", design, beta)
    residual = y_matrix - fitted
    sigma2 = np.einsum("fn,fn->f", residual, residual) / DF_RESID
    standard_error = np.sqrt(
        np.maximum(sigma2 * inverse_interaction_diag, 0.0)
    )
    t_value = np.divide(
        beta[:, 3],
        standard_error,
        out=np.zeros_like(standard_error),
        where=standard_error > np.finfo(float).eps,
    )
    p_value = 2.0 * stats.t.sf(np.abs(t_value), DF_RESID)
    return beta[:, 3], t_value, p_value


def build_preparations(
    data: dict[str, pd.DataFrame],
) -> tuple[list[ScreenPreparation], pd.DataFrame]:
    preparations: list[ScreenPreparation] = []
    combo_rows = []
    combo_index = 0
    for family in FAMILIES:
        frame = data[family]
        x_raw = frame[EEG_IDS].to_numpy(float)
        x_centered = x_raw - x_raw.mean(axis=0, keepdims=True)
        for outcome in OUTCOMES:
            y = frame[outcome].to_numpy(float)
            for moderator in MODERATORS:
                w = frame[moderator].to_numpy(float)
                ones = np.ones_like(x_centered)
                w_matrix = np.broadcast_to(w[:, None], x_centered.shape)
                reduced_design = np.stack(
                    [
                        ones.T,
                        x_centered.T,
                        w_matrix.T,
                    ],
                    axis=2,
                )
                full_design = np.stack(
                    [
                        ones.T,
                        x_centered.T,
                        w_matrix.T,
                        (x_centered * w_matrix).T,
                    ],
                    axis=2,
                )
                if np.any(
                    np.linalg.matrix_rank(full_design) != 4
                ):
                    raise np.linalg.LinAlgError(
                        f"{family}/{outcome}/{moderator}: diseño sin rango."
                    )
                _, reduced_pinv = prepare_batched_design(reduced_design)
                full_inverse, full_pinv = prepare_batched_design(full_design)
                y_matrix = np.broadcast_to(y, (64, len(y)))
                reduced_beta = np.einsum(
                    "fkn,fn->fk", reduced_pinv, y_matrix
                )
                reduced_fitted = np.einsum(
                    "fnk,fk->fn", reduced_design, reduced_beta
                )
                reduced_residuals = y_matrix - reduced_fitted
                observed_beta, observed_t, observed_p = fit_batched(
                    y_matrix,
                    full_design,
                    full_pinv,
                    full_inverse[:, 3, 3],
                )
                preparations.append(
                    ScreenPreparation(
                        family=family,
                        outcome=outcome,
                        moderator=moderator,
                        combo_index=combo_index,
                        x_raw=x_raw,
                        y=y,
                        w=w,
                        reduced_fitted=reduced_fitted,
                        reduced_residuals=reduced_residuals,
                        full_design=full_design,
                        full_pinv=full_pinv,
                        full_inv_interaction_diag=full_inverse[:, 3, 3],
                        observed_beta=observed_beta,
                        observed_t=observed_t,
                        observed_p=observed_p,
                    )
                )
                combo_rows.append(
                    {
                        "combo_index": combo_index,
                        "analysis_family": family,
                        "analysis_label": FAMILY_LABELS[family],
                        "outcome": outcome,
                        "moderator_id": moderator,
                    }
                )
                combo_index += 1
    if combo_index != 30:
        raise ValueError(f"Se prepararon {combo_index} combinaciones.")
    return preparations, pd.DataFrame(combo_rows)


def validate_observed_screen(
    preparations: list[ScreenPreparation], comparison: pd.DataFrame
) -> tuple[float, float]:
    maximum_p_error = 0.0
    maximum_beta_error = 0.0
    indexed = comparison.set_index(
        ["analysis_family", "outcome", "moderator_id", "eeg_id"]
    )
    for preparation in preparations:
        for eeg_index, eeg_id in enumerate(EEG_IDS):
            stored = indexed.loc[
                (
                    preparation.family,
                    preparation.outcome,
                    preparation.moderator,
                    eeg_id,
                )
            ]
            maximum_p_error = max(
                maximum_p_error,
                abs(float(stored["p_raw"]) - preparation.observed_p[eeg_index]),
            )
            maximum_beta_error = max(
                maximum_beta_error,
                abs(
                    float(stored["beta_interaction"])
                    - preparation.observed_beta[eeg_index]
                ),
            )
    if maximum_p_error > 1e-10 or maximum_beta_error > 1e-10:
        raise ValueError(
            "El cribado vectorizado no reproduce los modelos originales: "
            f"error p={maximum_p_error}, beta={maximum_beta_error}."
        )
    return maximum_p_error, maximum_beta_error


def fit_single_network(
    y: np.ndarray,
    x_raw: np.ndarray,
    w: np.ndarray,
    residual_permutation: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    x = x_raw - x_raw.mean()
    reduced = np.column_stack([np.ones(len(y)), x, w])
    full = np.column_stack([np.ones(len(y)), x, w, x * w])
    if np.linalg.matrix_rank(full) != 4:
        return {
            "beta": 0.0,
            "t": 0.0,
            "p": 1.0,
            "p_hc3": 1.0,
            "reduced_fitted": np.zeros_like(y),
            "reduced_residual": np.zeros_like(y),
            "full": full,
            "full_pinv": np.linalg.pinv(full),
            "inv_interaction_diag": math.inf,
        }
    reduced_beta = np.linalg.lstsq(reduced, y, rcond=None)[0]
    reduced_fitted = reduced @ reduced_beta
    reduced_residual = y - reduced_fitted
    if residual_permutation is None:
        y_used = y
    else:
        y_used = reduced_fitted + reduced_residual[residual_permutation]
    inverse = np.linalg.inv(full.T @ full)
    pseudo_inverse = inverse @ full.T
    beta = pseudo_inverse @ y_used
    residual = y_used - full @ beta
    sigma2 = float(residual @ residual) / DF_RESID
    standard_error = math.sqrt(max(sigma2 * inverse[3, 3], 0.0))
    t_value = (
        float(beta[3] / standard_error)
        if standard_error > np.finfo(float).eps
        else 0.0
    )
    p_value = float(2.0 * stats.t.sf(abs(t_value), DF_RESID))
    p_hc3 = float("nan")
    if residual_permutation is None:
        robust = sm.OLS(y, full).fit().get_robustcov_results(cov_type="HC3")
        p_hc3 = float(robust.pvalues[3])
    return {
        "beta": float(beta[3]),
        "t": t_value,
        "p": p_value,
        "p_hc3": p_hc3,
        "reduced_fitted": reduced_fitted,
        "reduced_residual": reduced_residual,
        "full": full,
        "full_pinv": pseudo_inverse,
        "inv_interaction_diag": float(inverse[3, 3]),
    }


def observed_networks(
    preparations: list[ScreenPreparation],
    combo_table: pd.DataFrame,
    mapping: pd.DataFrame,
    screen_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    network_rows = []
    feature_rows = []
    observed_abs_t = np.zeros(30, dtype=float)
    mapping_indexed = mapping.set_index("eeg_id")
    for preparation in preparations:
        selected = preparation.observed_p < screen_alpha
        selected_ids = [
            eeg_id for eeg_id, keep in zip(EEG_IDS, selected) if keep
        ]
        if selected_ids:
            score = preparation.x_raw[:, selected].mean(axis=1)
            fit = fit_single_network(
                preparation.y, score, preparation.w, residual_permutation=None
            )
            observed_abs_t[preparation.combo_index] = abs(float(fit["t"]))
        else:
            fit = {
                "beta": 0.0,
                "t": 0.0,
                "p": 1.0,
                "p_hc3": 1.0,
            }
        network_rows.append(
            {
                "combo_index": preparation.combo_index,
                "analysis_family": preparation.family,
                "analysis_label": FAMILY_LABELS[preparation.family],
                "outcome": preparation.outcome,
                "moderator_id": preparation.moderator,
                "observed_n_selected_features": int(selected.sum()),
                "observed_selected_eeg_ids": " | ".join(selected_ids),
                "observed_beta_interaction": float(fit["beta"]),
                "observed_t_interaction": float(fit["t"]),
                "observed_p_naive": float(fit["p"]),
                "observed_p_hc3_naive": float(fit["p_hc3"]),
            }
        )
        for eeg_index in np.flatnonzero(selected):
            eeg_id = EEG_IDS[eeg_index]
            detail = mapping_indexed.loc[eeg_id]
            feature_rows.append(
                {
                    "combo_index": preparation.combo_index,
                    "analysis_family": preparation.family,
                    "outcome": preparation.outcome,
                    "moderator_id": preparation.moderator,
                    "eeg_id": eeg_id,
                    "channel": detail["channel"],
                    "band": detail["band"],
                    "screen_beta_interaction": preparation.observed_beta[eeg_index],
                    "screen_t_interaction": preparation.observed_t[eeg_index],
                    "screen_p_raw": preparation.observed_p[eeg_index],
                }
            )
    networks = pd.DataFrame(network_rows).merge(
        combo_table,
        on=[
            "combo_index",
            "analysis_family",
            "analysis_label",
            "outcome",
            "moderator_id",
        ],
        validate="one_to_one",
    )
    features = pd.DataFrame(feature_rows)
    if len(features) != 51:
        raise ValueError(
            f"Se esperaban 51 selecciones observadas; hay {len(features)}."
        )
    if networks.groupby("analysis_family")[
        "observed_n_selected_features"
    ].sum().to_dict() != {
        "OC": 31,
        "OA": 2,
        "DELTA_OC_MINUS_OA": 18,
    }:
        raise ValueError("No se reprodujeron los conteos 31/2/18.")
    return networks, features, observed_abs_t


def generate_stratified_permutations(
    data: dict[str, pd.DataFrame], permutations: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_keys = rng.random((permutations, 25))
    output = np.empty((permutations, 5, 25), dtype=np.int16)
    reference = data["OC"]
    for moderator_index, moderator in enumerate(MODERATORS):
        w = reference[moderator].to_numpy(int)
        for permutation_index in range(permutations):
            permutation = np.arange(25, dtype=np.int16)
            keys = random_keys[permutation_index]
            for level in [0, 1]:
                members = np.flatnonzero(w == level)
                ordered = members[
                    np.argsort(keys[members], kind="mergesort")
                ]
                permutation[members] = ordered
            output[permutation_index, moderator_index] = permutation
    return output


def permutation_network_t(
    preparation: ScreenPreparation,
    selected: np.ndarray,
    permutation: np.ndarray,
) -> float:
    if not selected.any():
        return 0.0
    score = preparation.x_raw[:, selected].mean(axis=1)
    fit = fit_single_network(
        preparation.y,
        score,
        preparation.w,
        residual_permutation=permutation,
    )
    return abs(float(fit["t"]))


def run_permutations(
    preparations: list[ScreenPreparation],
    permutations: np.ndarray,
    output: Path,
    screen_alpha: float,
    chunk_size: int,
    checkpoint_every: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = permutations.shape[0]
    null_abs_t = np.zeros((count, 30), dtype=np.float32)
    null_selected = np.zeros((count, 30), dtype=np.uint8)
    selection_frequency = np.zeros((30, 64), dtype=np.int64)
    checkpoint_path = output / "checkpoint_selective_permutation.npz"
    start_index = 0
    if checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path)
        checkpoint_count = int(checkpoint["permutation_count"])
        if checkpoint_count != count:
            raise ValueError(
                "El checkpoint pertenece a otro número de permutaciones."
            )
        start_index = int(checkpoint["completed"])
        null_abs_t[:start_index] = checkpoint["null_abs_t"][:start_index]
        null_selected[:start_index] = checkpoint["null_selected"][:start_index]
        selection_frequency = checkpoint["selection_frequency"].astype(np.int64)
        logger.info("Reanudando desde la permutación %d.", start_index)

    started = time.perf_counter()
    moderator_to_index = {
        moderator: index for index, moderator in enumerate(MODERATORS)
    }
    for chunk_start in range(start_index, count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, count)
        chunk_indices = np.arange(chunk_start, chunk_end)
        chunk_length = chunk_end - chunk_start
        for preparation in preparations:
            moderator_index = moderator_to_index[preparation.moderator]
            permutation_chunk = permutations[
                chunk_start:chunk_end, moderator_index, :
            ]
            permuted_residuals = preparation.reduced_residuals[
                :, permutation_chunk
            ].transpose(1, 0, 2)
            pseudo_y = (
                preparation.reduced_fitted[None, :, :] + permuted_residuals
            )
            beta = np.einsum(
                "fkn,cfn->cfk", preparation.full_pinv, pseudo_y
            )
            fitted = np.einsum(
                "fnk,cfk->cfn", preparation.full_design, beta
            )
            residual = pseudo_y - fitted
            sigma2 = (
                np.einsum("cfn,cfn->cf", residual, residual) / DF_RESID
            )
            standard_error = np.sqrt(
                np.maximum(
                    sigma2
                    * preparation.full_inv_interaction_diag[None, :],
                    0.0,
                )
            )
            t_screen = np.divide(
                beta[:, :, 3],
                standard_error,
                out=np.zeros_like(standard_error),
                where=standard_error > np.finfo(float).eps,
            )
            p_screen = 2.0 * stats.t.sf(np.abs(t_screen), DF_RESID)
            selected_matrix = p_screen < screen_alpha
            null_selected[
                chunk_start:chunk_end, preparation.combo_index
            ] = selected_matrix.sum(axis=1).astype(np.uint8)
            selection_frequency[preparation.combo_index] += (
                selected_matrix.sum(axis=0).astype(np.int64)
            )
            for local_index in range(chunk_length):
                null_abs_t[
                    chunk_indices[local_index], preparation.combo_index
                ] = permutation_network_t(
                    preparation,
                    selected_matrix[local_index],
                    permutation_chunk[local_index],
                )

        completed = chunk_end
        elapsed = time.perf_counter() - started
        processed_this_run = max(1, completed - start_index)
        rate = processed_this_run / max(elapsed, np.finfo(float).eps)
        remaining_seconds = (count - completed) / max(
            rate, np.finfo(float).eps
        )
        progress = {
            "status": "running" if completed < count else "completed",
            "completed": completed,
            "total": count,
            "percent": 100.0 * completed / count,
            "iterations_per_second": rate,
            "elapsed_seconds_this_run": elapsed,
            "estimated_remaining_seconds": remaining_seconds,
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        atomic_json(output / "progress.json", progress)
        if (
            completed % checkpoint_every == 0
            or completed == count
        ):
            atomic_npz(
                checkpoint_path,
                permutation_count=np.array(count, dtype=np.int64),
                completed=np.array(completed, dtype=np.int64),
                null_abs_t=null_abs_t,
                null_selected=null_selected,
                selection_frequency=selection_frequency,
            )
        if completed % max(chunk_size, 500) == 0 or completed == count:
            logger.info(
                "Permutaciones %d/%d (%.1f%%); %.2f iter/s; ETA %.1f min.",
                completed,
                count,
                100.0 * completed / count,
                rate,
                remaining_seconds / 60.0,
            )
    return null_abs_t, null_selected, selection_frequency


def empirical_p(null_values: np.ndarray, observed: float) -> tuple[int, float]:
    exceedances = int(np.count_nonzero(null_values >= observed - 1e-12))
    p_value = (exceedances + 1.0) / (len(null_values) + 1.0)
    return exceedances, p_value


def monte_carlo_interval(
    exceedances: int, permutations: int, confidence: float = 0.95
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    successes = exceedances + 1
    trials = permutations + 1
    lower = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(
            stats.beta.ppf(
                1.0 - alpha / 2.0, successes + 1, trials - successes
            )
        )
    )
    return lower, upper


def add_inference(
    networks: pd.DataFrame,
    observed_abs_t: np.ndarray,
    null_abs_t: np.ndarray,
    null_selected: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = networks.sort_values("combo_index").reset_index(drop=True).copy()
    family_indices = {
        family: results.index[
            results["analysis_family"].eq(family)
        ].to_numpy()
        for family in FAMILIES
    }
    global_max = null_abs_t.max(axis=1)
    family_max = {
        family: null_abs_t[:, indices].max(axis=1)
        for family, indices in family_indices.items()
    }
    inference_rows = []
    permutations = len(null_abs_t)
    for row in results.itertuples(index=False):
        index = int(row.combo_index)
        if row.observed_n_selected_features > 0:
            selective_exceed, selective_p = empirical_p(
                null_abs_t[:, index], observed_abs_t[index]
            )
            family_exceed, family_p = empirical_p(
                family_max[row.analysis_family], observed_abs_t[index]
            )
            global_exceed, global_p = empirical_p(
                global_max, observed_abs_t[index]
            )
            size_exceed, size_p = empirical_p(
                null_selected[:, index].astype(float),
                float(row.observed_n_selected_features),
            )
        else:
            selective_exceed = family_exceed = global_exceed = permutations
            size_exceed = permutations
            selective_p = family_p = global_p = size_p = 1.0
        lower, upper = monte_carlo_interval(
            selective_exceed, permutations
        )
        inference_rows.append(
            {
                "combo_index": index,
                "selective_exceedances": selective_exceed,
                "p_selective_pipeline": selective_p,
                "p_selective_mc95_low": lower,
                "p_selective_mc95_high": upper,
                "maxT_family_exceedances": family_exceed,
                "p_maxT_family": family_p,
                "maxT_global_exceedances": global_exceed,
                "p_maxT_global": global_p,
                "network_size_exceedances": size_exceed,
                "p_network_size": size_p,
                "null_mean_selected_features": float(
                    null_selected[:, index].mean()
                ),
                "null_median_selected_features": float(
                    np.median(null_selected[:, index])
                ),
                "null_q95_selected_features": float(
                    np.quantile(null_selected[:, index], 0.95)
                ),
            }
        )
    inference = pd.DataFrame(inference_rows)
    results = results.merge(
        inference, on="combo_index", validate="one_to_one"
    )
    results["p_FDR_selective_within_family_10"] = np.nan
    results["p_BY_selective_within_family_10"] = np.nan
    for family, indices in family_indices.items():
        values = results.loc[indices, "p_selective_pipeline"].to_numpy(float)
        results.loc[indices, "p_FDR_selective_within_family_10"] = (
            multipletests(values, method="fdr_bh")[1]
        )
        results.loc[indices, "p_BY_selective_within_family_10"] = (
            multipletests(values, method="fdr_by")[1]
        )
    values = results["p_selective_pipeline"].to_numpy(float)
    results["p_FDR_selective_global_30"] = multipletests(
        values, method="fdr_bh"
    )[1]
    results["p_BY_selective_global_30"] = multipletests(
        values, method="fdr_by"
    )[1]
    results["hit_FDR_selective_global_0_10"] = (
        results["p_FDR_selective_global_30"] < 0.10
    )
    results["hit_maxT_global_0_05"] = results["p_maxT_global"] < 0.05

    family_rows = []
    for family, indices in family_indices.items():
        observed_family_max = float(observed_abs_t[indices].max())
        exceedances, p_value = empirical_p(
            family_max[family], observed_family_max
        )
        family_rows.append(
            {
                "analysis_family": family,
                "analysis_label": FAMILY_LABELS[family],
                "observed_max_abs_t": observed_family_max,
                "exceedances": exceedances,
                "p_omnibus_maxT_family": p_value,
                "observed_total_selected_features": int(
                    results.loc[
                        indices, "observed_n_selected_features"
                    ].sum()
                ),
                "observed_networks": int(
                    (
                        results.loc[
                            indices, "observed_n_selected_features"
                        ]
                        > 0
                    ).sum()
                ),
            }
        )
    global_observed = float(observed_abs_t.max())
    global_exceedances, global_p = empirical_p(global_max, global_observed)
    family_rows.append(
        {
            "analysis_family": "GLOBAL_30",
            "analysis_label": "Tres familias, 30 combinaciones",
            "observed_max_abs_t": global_observed,
            "exceedances": global_exceedances,
            "p_omnibus_maxT_family": global_p,
            "observed_total_selected_features": int(
                results["observed_n_selected_features"].sum()
            ),
            "observed_networks": int(
                (results["observed_n_selected_features"] > 0).sum()
            ),
        }
    )
    return results, pd.DataFrame(family_rows)


def build_null_summary(
    null_abs_t: np.ndarray,
    null_selected: np.ndarray,
    combo_table: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "permutation": np.arange(1, len(null_abs_t) + 1),
            "global_max_abs_t_30": null_abs_t.max(axis=1),
            "total_selected_features_1920": null_selected.sum(axis=1),
            "networks_formed_30": (null_selected > 0).sum(axis=1),
        }
    )
    for family in FAMILIES:
        indices = combo_table.index[
            combo_table["analysis_family"].eq(family)
        ].to_numpy()
        output[f"{family}_max_abs_t_10"] = null_abs_t[:, indices].max(axis=1)
        output[f"{family}_selected_features_640"] = null_selected[
            :, indices
        ].sum(axis=1)
        output[f"{family}_networks_10"] = (
            null_selected[:, indices] > 0
        ).sum(axis=1)
    return output


def build_selection_frequency(
    selection_counts: np.ndarray,
    combo_table: pd.DataFrame,
    mapping: pd.DataFrame,
    permutations: int,
) -> pd.DataFrame:
    rows = []
    mapping_index = mapping.set_index("eeg_id")
    for combo in combo_table.itertuples(index=False):
        for eeg_index, eeg_id in enumerate(EEG_IDS):
            detail = mapping_index.loc[eeg_id]
            rows.append(
                {
                    "combo_index": combo.combo_index,
                    "analysis_family": combo.analysis_family,
                    "outcome": combo.outcome,
                    "moderator_id": combo.moderator_id,
                    "eeg_id": eeg_id,
                    "channel": detail["channel"],
                    "band": detail["band"],
                    "null_selection_count": int(
                        selection_counts[combo.combo_index, eeg_index]
                    ),
                    "null_selection_frequency": float(
                        selection_counts[combo.combo_index, eeg_index]
                        / permutations
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_global_null(
    null_abs_t: np.ndarray,
    observed_abs_t: np.ndarray,
    output: Path,
) -> None:
    observed = float(observed_abs_t.max())
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(
        null_abs_t.max(axis=1),
        bins=50,
        color="#4472C4",
        alpha=0.82,
        edgecolor="white",
    )
    axis.axvline(
        observed,
        color="#C00000",
        linewidth=2.2,
        label=f"Máximo observado |t|={observed:.3f}",
    )
    axis.set_xlabel("Máximo |t| de las 30 firmas selectivas")
    axis.set_ylabel("Permutaciones")
    axis.set_title("Distribución nula global del pipeline completo")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "global_maxT_null.png", dpi=180)
    plt.close(figure)


def plot_empirical_p(results: pd.DataFrame, output: Path) -> None:
    subset = results.loc[
        results["observed_n_selected_features"] > 0
    ].copy()
    subset["label"] = (
        subset["analysis_family"]
        + " | "
        + subset["outcome"]
        + "×"
        + subset["moderator_id"]
    )
    subset = subset.sort_values("p_selective_pipeline", ascending=True)
    figure_height = max(4.8, 0.52 * len(subset) + 1.8)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    positions = np.arange(len(subset))
    axis.barh(
        positions,
        -np.log10(subset["p_selective_pipeline"].to_numpy()),
        color="#70AD47",
        label="p selectiva",
    )
    axis.scatter(
        -np.log10(subset["p_maxT_global"].to_numpy()),
        positions,
        color="#C00000",
        marker="D",
        s=34,
        label="p maxT global",
        zorder=3,
    )
    axis.axvline(-math.log10(0.05), color="#666666", linestyle="--")
    axis.set_yticks(positions, subset["label"].tolist())
    axis.invert_yaxis()
    axis.set_xlabel("−log10(p)")
    axis.set_title("Inferencia después de repetir selección y agregación")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "selective_empirical_p.png", dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    results: pd.DataFrame,
    omnibus: pd.DataFrame,
    permutations: int,
    seed: int,
    hashes_before: dict[str, str],
    hashes_preserved: bool,
    elapsed_seconds: float,
    screen_p_error: float,
    screen_beta_error: float,
) -> None:
    observed = results.loc[
        results["observed_n_selected_features"] > 0
    ].sort_values("p_selective_pipeline")
    lines = [
        "INFORME — PERMUTACIÓN SELECTIVA COMPLETA",
        "=" * 72,
        f"Fecha: {datetime.now().astimezone().isoformat()}",
        f"Permutaciones: {permutations:,}",
        f"Semilla: {seed}",
        f"Tiempo: {elapsed_seconds / 60.0:.2f} minutos",
        "",
        "PIPELINE FIJADO ANTES DE LA EJECUCIÓN",
        "- 30 combinaciones: 3 familias × 2 outcomes × 5 moderadores.",
        "- Cada combinación criba 64 EEG con p OLS bilateral < .05.",
        "- Firma: media de los EEG seleccionados, en dB y con pesos iguales.",
        "- Estadístico final: |t| de firma×W.",
        "- Freedman–Lane: residuos del modelo reducido, permutados por sujeto",
        "  dentro de W=0/W=1.",
        "- La misma permutación por W se aplica a OC, OA, OC−OA y outcomes.",
        "- Correcciones: BH/BY entre 30; maxT por familia y global.",
        "",
        "REPRODUCCIÓN DE LOS MODELOS ORIGINALES",
        f"- Error máximo p: {screen_p_error:.3e}.",
        f"- Error máximo beta: {screen_beta_error:.3e}.",
        "- Selecciones observadas: OC=31; OA=2; OC−OA=18.",
        "",
        "OMNIBUS maxT",
    ]
    for row in omnibus.itertuples(index=False):
        lines.append(
            f"- {row.analysis_family}: max |t|={row.observed_max_abs_t:.6g}; "
            f"p={row.p_omnibus_maxT_family:.6g}; "
            f"excedencias={row.exceedances}/{permutations}."
        )
    lines.extend(["", "FIRMAS OBSERVADAS"])
    for row in observed.itertuples(index=False):
        lines.append(
            f"- {row.analysis_family}/{row.outcome}×{row.moderator_id}: "
            f"k={row.observed_n_selected_features}; "
            f"t={row.observed_t_interaction:.6g}; "
            f"p ingenua={row.observed_p_naive:.6g}; "
            f"p selectiva={row.p_selective_pipeline:.6g}; "
            f"q30={row.p_FDR_selective_global_30:.6g}; "
            f"p maxT global={row.p_maxT_global:.6g}."
        )
    lines.extend(
        [
            "",
            "DECISIÓN",
            f"- Hits BH-FDR q30<.10: "
            f"{int(results['hit_FDR_selective_global_0_10'].sum())}.",
            f"- Hits maxT global p<.05: "
            f"{int(results['hit_maxT_global_0_05'].sum())}.",
            "- La inferencia incorpora la selección de canal×banda y la creación",
            "  de la firma; las p ingenuas no se interpretan como confirmatorias.",
            "- Resultados asociativos. No implican causalidad, red funcional ni",
            "  biomarcador.",
            "",
            "INTEGRIDAD",
            f"- Resultados previos preservados por hash: {hashes_preserved}.",
            f"- Archivos fuente verificados: {len(hashes_before)}.",
        ]
    )
    (output / "informe_permutacion_selectiva.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_file_manifest(output: Path) -> None:
    rows = []
    excluded = {"manifest_archivos.csv"}
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
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
    if not 0.0 < args.screen_alpha < 1.0:
        raise ValueError("--screen-alpha debe estar entre 0 y 1.")
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output)
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    logger.info("Inicio del pipeline selectivo.")
    logger.info(
        "Permutaciones=%d; semilla=%d; alpha cribado=%.3f.",
        args.permutations,
        args.seed,
        args.screen_alpha,
    )

    data, comparison, mapping, hashes_before = load_inputs(args.project)
    logger.info("Insumos validados; hashes=%d.", len(hashes_before))
    preparations, combo_table = build_preparations(data)
    p_error, beta_error = validate_observed_screen(
        preparations, comparison
    )
    logger.info(
        "Cribado reproducido: error p=%.3e; beta=%.3e.",
        p_error,
        beta_error,
    )
    networks, selected_features, observed_abs_t = observed_networks(
        preparations, combo_table, mapping, args.screen_alpha
    )
    logger.info(
        "Selección observada: firmas=%d; características=%d.",
        int((networks["observed_n_selected_features"] > 0).sum()),
        len(selected_features),
    )

    permutation_indices = generate_stratified_permutations(
        data, args.permutations, args.seed
    )
    atomic_npz(
        args.output / "permutation_indices.npz",
        permutation_indices=permutation_indices,
        seed=np.array(args.seed, dtype=np.int64),
    )
    null_abs_t, null_selected, selection_frequency = run_permutations(
        preparations,
        permutation_indices,
        args.output,
        args.screen_alpha,
        args.chunk_size,
        args.checkpoint_every,
        logger,
    )
    networks, omnibus = add_inference(
        networks, observed_abs_t, null_abs_t, null_selected
    )
    null_summary = build_null_summary(
        null_abs_t, null_selected, combo_table
    )
    frequency_table = build_selection_frequency(
        selection_frequency, combo_table, mapping, args.permutations
    )

    networks.to_csv(
        args.output / "resultados_30_firmas_selectivas.csv", index=False
    )
    selected_features.to_csv(
        args.output / "caracteristicas_51_seleccionadas_observadas.csv",
        index=False,
    )
    omnibus.to_csv(
        args.output / "resultados_omnibus_maxT.csv", index=False
    )
    null_summary.to_csv(
        args.output
        / f"distribucion_nula_resumen_{args.permutations}.csv",
        index=False,
    )
    frequency_table.to_csv(
        args.output / "frecuencia_seleccion_nula_1920.csv", index=False
    )
    atomic_npz(
        args.output / "distribucion_nula_arrays.npz",
        null_abs_t=null_abs_t,
        null_selected=null_selected,
        observed_abs_t=observed_abs_t,
    )
    plot_global_null(null_abs_t, observed_abs_t, args.output)
    plot_empirical_p(networks, args.output)

    hashes_after = {
        path: sha256_file(Path(path)) for path in hashes_before
    }
    hashes_preserved = hashes_before == hashes_after
    if not hashes_preserved:
        raise RuntimeError("Cambió al menos un resultado previo.")
    elapsed_seconds = time.perf_counter() - started
    write_report(
        args.output,
        networks,
        omnibus,
        args.permutations,
        args.seed,
        hashes_before,
        hashes_preserved,
        elapsed_seconds,
        p_error,
        beta_error,
    )
    manifest = {
        "analysis": "selective_permutation_full_pipeline",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "project": str(args.project),
        "output": str(args.output),
        "permutations": args.permutations,
        "seed": args.seed,
        "screen_alpha": args.screen_alpha,
        "permutation_method": (
            "Freedman-Lane residual permutation within binary moderator strata"
        ),
        "dependency_preservation": (
            "same subject permutation within moderator across OC, OA, delta "
            "and both outcomes"
        ),
        "screen_models": 1920,
        "combination_family": 30,
        "observed_selected_features": int(
            networks["observed_n_selected_features"].sum()
        ),
        "observed_networks": int(
            (networks["observed_n_selected_features"] > 0).sum()
        ),
        "screen_reproduction_max_p_error": p_error,
        "screen_reproduction_max_beta_error": beta_error,
        "fdr_global_hits_q_lt_0_10": int(
            networks["hit_FDR_selective_global_0_10"].sum()
        ),
        "maxT_global_hits_p_lt_0_05": int(
            networks["hit_maxT_global_0_05"].sum()
        ),
        "prior_hashes_before": hashes_before,
        "prior_hashes_after": hashes_after,
        "prior_results_preserved": hashes_preserved,
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
            "completed": args.permutations,
            "total": args.permutations,
            "percent": 100.0,
            "elapsed_seconds": elapsed_seconds,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )
    logger.info("Pipeline concluido en %.2f minutos.", elapsed_seconds / 60.0)
    logger.info(
        "Hits: q30<.10=%d; maxT global p<.05=%d.",
        int(networks["hit_FDR_selective_global_0_10"].sum()),
        int(networks["hit_maxT_global_0_05"].sum()),
    )
    logger.info("Resultados previos preservados: %s.", hashes_preserved)
    logging.shutdown()
    write_file_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
