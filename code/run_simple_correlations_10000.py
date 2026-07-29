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


OUTCOMES = ["BAI", "MSPSS", "DISCREPANCY_Z_BAI_MINUS_MSPSS"]
REPRESENTATIONS = ["OC", "OA", "DELTA_OA_MINUS_OC"]
EEG_IDS = [f"EEG{i}" for i in range(1, 65)]
N_SUBJECTS = 25
DF_CORRELATION = N_SUBJECTS - 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlaciones lineales EEG con 10,000 permutaciones."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--chunk-size", type=int, default=500)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("simple_correlations")
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


def zscore_sample(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std(ddof=1)


def load_data(
    project: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    paths = {
        "OC": project / "output" / "analysis_dataset_25.csv",
        "OA": (
            project
            / "output_oa_oc_delta_comparison"
            / "eyes_open"
            / "analysis_dataset_25.csv"
        ),
        "mapping": project / "output" / "eeg_variable_mapping.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    oc = pd.read_csv(paths["OC"]).sort_values("subject_id").reset_index(drop=True)
    oa = pd.read_csv(paths["OA"]).sort_values("subject_id").reset_index(drop=True)
    required = {"subject_id", "BAI", "MSPSS", *EEG_IDS}
    for label, frame in {"OC": oc, "OA": oa}.items():
        if len(frame) != N_SUBJECTS or frame["subject_id"].nunique() != N_SUBJECTS:
            raise ValueError(f"{label}: unidad sujeto inválida.")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: faltan {sorted(missing)}.")
        frame[["BAI", "MSPSS", *EEG_IDS]] = frame[
            ["BAI", "MSPSS", *EEG_IDS]
        ].apply(pd.to_numeric, errors="raise")
    if not oc[["subject_id", "BAI", "MSPSS"]].equals(
        oa[["subject_id", "BAI", "MSPSS"]]
    ):
        raise ValueError("OC y OA no comparten sujetos/outcomes.")

    delta = oc[["subject_id", "BAI", "MSPSS"]].copy()
    delta[EEG_IDS] = oa[EEG_IDS].to_numpy(float) - oc[EEG_IDS].to_numpy(float)
    representations = {"OC": oc, "OA": oa, "DELTA_OA_MINUS_OC": delta}
    mapping = pd.read_csv(paths["mapping"])
    if len(mapping) != 64 or mapping["eeg_id"].tolist() != EEG_IDS:
        raise ValueError("Mapping EEG inválido.")
    hashes = {str(path): sha256_file(path) for path in paths.values()}
    for prior in [
        project
        / "output_oa_oc_delta_comparison"
        / "resultados_1920_comparados.csv",
        project
        / "output_selective_permutation_10000"
        / "resultados_30_firmas_selectivas.csv",
    ]:
        if prior.is_file():
            hashes[str(prior)] = sha256_file(prior)
    return representations, mapping, hashes


def build_outcomes(
    oc: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    bai = oc["BAI"].to_numpy(float)
    mspss = oc["MSPSS"].to_numpy(float)
    z_bai = zscore_sample(bai)
    z_mspss = zscore_sample(mspss)
    discrepancy = z_bai - z_mspss
    tolerance = 1e-12
    labels = np.where(
        discrepancy > tolerance,
        "BAI>MSPSS",
        np.where(discrepancy < -tolerance, "MSPSS>BAI", "BAI~MSPSS"),
    )
    table = pd.DataFrame(
        {
            "subject_id": oc["subject_id"],
            "BAI": bai,
            "MSPSS": mspss,
            "z_BAI": z_bai,
            "z_MSPSS": z_mspss,
            "discrepancy_z_BAI_minus_z_MSPSS": discrepancy,
            "discrepancy_group": labels,
        }
    )
    y_raw = np.column_stack([bai, mspss, discrepancy])
    y_standardized = np.column_stack(
        [zscore_sample(y_raw[:, index]) for index in range(3)]
    )
    return y_standardized, table


def parametric_p_from_r(r_value: np.ndarray) -> np.ndarray:
    denominator = np.maximum(1.0 - np.square(r_value), np.finfo(float).tiny)
    t_value = r_value * np.sqrt(DF_CORRELATION / denominator)
    return 2.0 * stats.t.sf(np.abs(t_value), DF_CORRELATION)


def generate_permutations(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutations = np.empty((count, N_SUBJECTS), dtype=np.int16)
    for index in range(count):
        permutations[index] = rng.permutation(N_SUBJECTS)
    return permutations


def compute_observed(
    y_standardized: np.ndarray,
    representations: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    standardized_eeg = {}
    mapping_index = mapping.set_index("eeg_id")
    for representation in REPRESENTATIONS:
        x_raw = representations[representation][EEG_IDS].to_numpy(float)
        x_standardized = np.column_stack(
            [zscore_sample(x_raw[:, index]) for index in range(64)]
        )
        standardized_eeg[representation] = x_standardized
        correlations = (
            y_standardized.T @ x_standardized / (N_SUBJECTS - 1)
        )
        p_values = parametric_p_from_r(correlations)
        for outcome_index, outcome in enumerate(OUTCOMES):
            for eeg_index, eeg_id in enumerate(EEG_IDS):
                detail = mapping_index.loc[eeg_id]
                rows.append(
                    {
                        "outcome": outcome,
                        "representation": representation,
                        "eeg_id": eeg_id,
                        "channel": detail["channel"],
                        "band": detail["band"],
                        "n": N_SUBJECTS,
                        "r_pearson": correlations[outcome_index, eeg_index],
                        "p_raw_parametric": p_values[
                            outcome_index, eeg_index
                        ],
                    }
                )
    return pd.DataFrame(rows), standardized_eeg


def run_permutations(
    y_standardized: np.ndarray,
    standardized_eeg: dict[str, np.ndarray],
    permutations: np.ndarray,
    chunk_size: int,
    output: Path,
    logger: logging.Logger,
) -> np.ndarray:
    count = len(permutations)
    null_abs_r = np.empty((count, 3, 3, 64), dtype=np.float32)
    started = time.perf_counter()
    for chunk_start in range(0, count, chunk_size):
        chunk_end = min(chunk_start + chunk_size, count)
        permutation_chunk = permutations[chunk_start:chunk_end]
        permuted_y = y_standardized[permutation_chunk]
        for representation_index, representation in enumerate(REPRESENTATIONS):
            correlations = np.einsum(
                "cno,nf->cof",
                permuted_y,
                standardized_eeg[representation],
            ) / (N_SUBJECTS - 1)
            null_abs_r[
                chunk_start:chunk_end, :, representation_index, :
            ] = np.abs(correlations).astype(np.float32)
        completed = chunk_end
        elapsed = time.perf_counter() - started
        rate = completed / max(elapsed, np.finfo(float).eps)
        remaining = (count - completed) / max(rate, np.finfo(float).eps)
        atomic_json(
            output / "progress.json",
            {
                "status": "running" if completed < count else "completed",
                "completed": completed,
                "total": count,
                "percent": completed / count * 100.0,
                "iterations_per_second": rate,
                "estimated_remaining_seconds": remaining,
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        if completed % 2_000 == 0 or completed == count:
            logger.info(
                "Permutaciones %d/%d (%.1f%%); %.1f iter/s.",
                completed,
                count,
                completed / count * 100.0,
                rate,
            )
    return null_abs_r


def empirical_p(
    null_values: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    exceedances = np.sum(
        null_values >= observed[None, ...] - 1e-12, axis=0
    )
    p_values = (exceedances + 1.0) / (len(null_values) + 1.0)
    return exceedances.astype(int), p_values


def add_inference(
    results: pd.DataFrame,
    null_abs_r: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = results.copy()
    result["outcome_index"] = result["outcome"].map(
        {name: index for index, name in enumerate(OUTCOMES)}
    )
    result["representation_index"] = result["representation"].map(
        {name: index for index, name in enumerate(REPRESENTATIONS)}
    )
    observed = np.zeros((3, 3, 64), dtype=float)
    for row in result.itertuples(index=False):
        eeg_index = int(row.eeg_id.replace("EEG", "")) - 1
        observed[
            row.outcome_index, row.representation_index, eeg_index
        ] = abs(row.r_pearson)
    exceedances, p_permutation = empirical_p(null_abs_r, observed)
    block_max = null_abs_r.max(axis=3)
    outcome_max = null_abs_r.max(axis=(2, 3))
    global_max = null_abs_r.max(axis=(1, 2, 3))

    p_perm_values = []
    p_max_block_values = []
    p_max_outcome_values = []
    p_max_global_values = []
    exceed_values = []
    for row in result.itertuples(index=False):
        eeg_index = int(row.eeg_id.replace("EEG", "")) - 1
        oi = row.outcome_index
        ri = row.representation_index
        obs = observed[oi, ri, eeg_index]
        p_perm_values.append(p_permutation[oi, ri, eeg_index])
        exceed_values.append(exceedances[oi, ri, eeg_index])
        p_max_block_values.append(
            (np.count_nonzero(block_max[:, oi, ri] >= obs - 1e-12) + 1)
            / (len(null_abs_r) + 1)
        )
        p_max_outcome_values.append(
            (np.count_nonzero(outcome_max[:, oi] >= obs - 1e-12) + 1)
            / (len(null_abs_r) + 1)
        )
        p_max_global_values.append(
            (np.count_nonzero(global_max >= obs - 1e-12) + 1)
            / (len(null_abs_r) + 1)
        )
    result["permutation_exceedances"] = exceed_values
    result["p_permutation"] = p_perm_values
    result["p_maxT_block64"] = p_max_block_values
    result["p_maxT_outcome192"] = p_max_outcome_values
    result["p_maxT_global576"] = p_max_global_values

    fdr_columns = [
        ("p_raw_parametric", "raw"),
        ("p_permutation", "permutation"),
    ]
    for p_column, label in fdr_columns:
        result[f"q_BH_{label}_block64"] = np.nan
        for _, indices in result.groupby(
            ["outcome", "representation"], sort=False
        ).groups.items():
            result.loc[indices, f"q_BH_{label}_block64"] = multipletests(
                result.loc[indices, p_column], method="fdr_bh"
            )[1]
        result[f"q_BH_{label}_outcome192"] = np.nan
        for _, indices in result.groupby("outcome", sort=False).groups.items():
            result.loc[indices, f"q_BH_{label}_outcome192"] = multipletests(
                result.loc[indices, p_column], method="fdr_bh"
            )[1]
        result[f"q_BH_{label}_global576"] = multipletests(
            result[p_column], method="fdr_bh"
        )[1]
    result["q_BY_permutation_global576"] = multipletests(
        result["p_permutation"], method="fdr_by"
    )[1]

    summaries = []
    for (outcome, representation), block in result.groupby(
        ["outcome", "representation"], sort=False
    ):
        summaries.append(
            {
                "outcome": outcome,
                "representation": representation,
                "models": len(block),
                "naive_p_lt_0_05": int(
                    (block["p_raw_parametric"] < 0.05).sum()
                ),
                "min_abs_r": float(block["r_pearson"].abs().min()),
                "max_abs_r": float(block["r_pearson"].abs().max()),
                "min_p_raw": float(block["p_raw_parametric"].min()),
                "min_p_permutation": float(block["p_permutation"].min()),
                "min_q_perm_block64": float(
                    block["q_BH_permutation_block64"].min()
                ),
                "hits_q_perm_block64_lt_0_05": int(
                    (block["q_BH_permutation_block64"] < 0.05).sum()
                ),
                "hits_q_perm_block64_lt_0_10": int(
                    (block["q_BH_permutation_block64"] < 0.10).sum()
                ),
                "min_p_maxT_block64": float(
                    block["p_maxT_block64"].min()
                ),
            }
        )
    result = result.drop(columns=["outcome_index", "representation_index"])
    return result, pd.DataFrame(summaries)


def build_frozen_networks(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = results.loc[results["p_raw_parametric"] < 0.05].copy()
    selected["network_id"] = (
        selected["outcome"] + "__" + selected["representation"]
    )
    group_sizes = selected.groupby("network_id")["eeg_id"].transform("size")
    selected["weight_sign_equal"] = (
        np.sign(selected["r_pearson"]) / group_sizes
    )
    network_summary = (
        selected.groupby(
            ["network_id", "outcome", "representation"], as_index=False
        )
        .agg(
            n_features=("eeg_id", "size"),
            eeg_ids=("eeg_id", lambda values: " | ".join(values)),
            channels=("channel", lambda values: " | ".join(values)),
            bands=("band", lambda values: " | ".join(values)),
            local_min_p=("p_raw_parametric", "min"),
            local_max_abs_r=("r_pearson", lambda values: np.max(np.abs(values))),
        )
    )
    return selected, network_summary


def build_null_summary(null_abs_r: np.ndarray) -> pd.DataFrame:
    rows = {
        "permutation": np.arange(1, len(null_abs_r) + 1),
        "global_max_abs_r_576": null_abs_r.max(axis=(1, 2, 3)),
    }
    for outcome_index, outcome in enumerate(OUTCOMES):
        rows[f"{outcome}_max_abs_r_192"] = null_abs_r[
            :, outcome_index
        ].max(axis=(1, 2))
        for representation_index, representation in enumerate(
            REPRESENTATIONS
        ):
            rows[f"{outcome}__{representation}_max_abs_r_64"] = (
                null_abs_r[:, outcome_index, representation_index].max(axis=1)
            )
    return pd.DataFrame(rows)


def plot_heatmaps(
    results: pd.DataFrame,
    output: Path,
) -> None:
    channel_order = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "F7",
        "F8",
        "C3",
        "C4",
        "T7",
        "T8",
        "P7",
        "P8",
        "P3",
        "P4",
        "O1",
        "O2",
    ]
    band_order = ["delta", "theta", "alpha", "beta"]
    figure, axes = plt.subplots(
        3, 3, figsize=(15, 15), constrained_layout=True
    )
    image = None
    for outcome_index, outcome in enumerate(OUTCOMES):
        for representation_index, representation in enumerate(
            REPRESENTATIONS
        ):
            axis = axes[outcome_index, representation_index]
            block = results.loc[
                results["outcome"].eq(outcome)
                & results["representation"].eq(representation)
            ]
            pivot = (
                block.pivot(index="channel", columns="band", values="r_pearson")
                .reindex(index=channel_order, columns=band_order)
            )
            image = axis.imshow(
                pivot.to_numpy(),
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                aspect="auto",
            )
            axis.set_xticks(range(4), band_order)
            axis.set_yticks(range(16), channel_order)
            axis.set_title(f"{outcome}\n{representation}")
            for row_index, channel in enumerate(channel_order):
                for column_index, band in enumerate(band_order):
                    value = float(pivot.loc[channel, band])
                    match = block.loc[
                        block["channel"].eq(channel)
                        & block["band"].eq(band)
                    ].iloc[0]
                    marker = "*" if match["p_raw_parametric"] < 0.05 else ""
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}{marker}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.45 else "black",
                    )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.65,
            label="r de Pearson",
        )
    figure.suptitle(
        "Correlaciones EEG–outcome (* p paramétrica ingenua < .05)",
        fontsize=16,
    )
    figure.savefig(output / "heatmaps_correlaciones_576.png", dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    results: pd.DataFrame,
    summaries: pd.DataFrame,
    networks: pd.DataFrame,
    discrepancy: pd.DataFrame,
    permutations: int,
    seed: int,
    prior_preserved: bool,
) -> None:
    top = results.sort_values("p_permutation").head(15)
    group_counts = discrepancy["discrepancy_group"].value_counts().to_dict()
    null_result = bool(
        (results["q_BH_permutation_global576"] >= 0.10).all()
        and (results["p_maxT_global576"] >= 0.05).all()
    )
    lemon_actionable = bool(null_result and len(networks) > 0)
    lines = [
        "INFORME — CORRELACIONES LINEALES EEG CON PERMUTACIÓN",
        "=" * 72,
        f"Fecha: {datetime.now().astimezone().isoformat()}",
        f"Permutaciones: {permutations:,}; semilla: {seed}.",
        "",
        "DISEÑO",
        "- Unidad: sujeto, n=25.",
        "- Outcomes: BAI, MSPSS y z(BAI)−z(MSPSS).",
        "- Representaciones: OC, OA y Δ=OA−OC.",
        "- 64 canal×banda por bloque; 576 correlaciones en total.",
        "- Permutación conjunta de los tres outcomes entre sujetos.",
        "- Inferencia: p empírica, BH por 64/192/576, BY global y maxT.",
        "",
        "DISCREPANCIA",
        f"- BAI>MSPSS: {group_counts.get('BAI>MSPSS', 0)}.",
        f"- MSPSS>BAI: {group_counts.get('MSPSS>BAI', 0)}.",
        f"- BAI~MSPSS: {group_counts.get('BAI~MSPSS', 0)}.",
        "",
        "RESUMEN POR BLOQUE",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"- {row.outcome}/{row.representation}: "
            f"p ingenua<.05={row.naive_p_lt_0_05}; "
            f"min p perm={row.min_p_permutation:.6g}; "
            f"min q64={row.min_q_perm_block64:.6g}; "
            f"hits q64<.05={row.hits_q_perm_block64_lt_0_05}; "
            f"min p maxT64={row.min_p_maxT_block64:.6g}."
        )
    lines.extend(["", "QUINCE RESULTADOS MÁS PEQUEÑOS"])
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.outcome}/{row.representation}/{row.eeg_id} "
            f"({row.channel}-{row.band}): r={row.r_pearson:.6g}; "
            f"p ingenua={row.p_raw_parametric:.6g}; "
            f"p perm={row.p_permutation:.6g}; "
            f"q576={row.q_BH_permutation_global576:.6g}; "
            f"p maxT576={row.p_maxT_global576:.6g}."
        )
    lines.extend(
        [
            "",
            "DECISIÓN",
            f"- Hits q perm global576<.05: "
            f"{int((results['q_BH_permutation_global576'] < 0.05).sum())}.",
            f"- Hits q perm global576<.10: "
            f"{int((results['q_BH_permutation_global576'] < 0.10).sum())}.",
            f"- Hits maxT global576<.05: "
            f"{int((results['p_maxT_global576'] < 0.05).sum())}.",
            f"- Redes nominales congeladas para posible LEMON: {len(networks)}.",
            f"- Resultado nulo activa la consideración de LEMON: {null_result}.",
            f"- Validación de red en LEMON accionable: {lemon_actionable}.",
            (
                "- Motivo de no ejecución: no existe una red local bajo la regla "
                "p ingenua<.05."
                if not lemon_actionable and len(networks) == 0
                else "- La validación externa requiere un script separado y la red congelada."
            ),
            "- Las etiquetas de discrepancia son descriptivas; la inferencia usa",
            "  el puntaje continuo.",
            "- Correlaciones asociativas; no implican causalidad o biomarcador.",
            "",
            "INTEGRIDAD",
            f"- Resultados previos preservados por hash: {prior_preserved}.",
        ]
    )
    (output / "informe_correlaciones_10000.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest(output: Path) -> None:
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
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output)
    started = time.perf_counter()
    started_at = datetime.now().astimezone()
    logger.info("Inicio de correlaciones simples.")
    representations, mapping, hashes_before = load_data(args.project)
    y_standardized, discrepancy = build_outcomes(representations["OC"])
    discrepancy.to_csv(
        args.output / "discrepancia_sujetos.csv", index=False
    )
    results, standardized_eeg = compute_observed(
        y_standardized, representations, mapping
    )
    logger.info(
        "Observado: modelos=%d; p ingenua<.05=%d.",
        len(results),
        int((results["p_raw_parametric"] < 0.05).sum()),
    )
    permutations = generate_permutations(args.permutations, args.seed)
    atomic_npz(
        args.output / "permutation_indices.npz",
        permutation_indices=permutations,
        seed=np.array(args.seed, dtype=np.int64),
    )
    null_abs_r = run_permutations(
        y_standardized,
        standardized_eeg,
        permutations,
        args.chunk_size,
        args.output,
        logger,
    )
    results, summaries = add_inference(results, null_abs_r)
    selected, networks = build_frozen_networks(results)
    null_summary = build_null_summary(null_abs_r)
    results.to_csv(
        args.output / "resultados_correlaciones_576.csv", index=False
    )
    summaries.to_csv(
        args.output / "resumen_bloques_9.csv", index=False
    )
    selected.to_csv(
        args.output / "resultados_p_ingenua_lt_0_05.csv", index=False
    )
    networks.to_csv(
        args.output / "redes_nominales_congeladas.csv", index=False
    )
    null_summary.to_csv(
        args.output / "distribucion_nula_resumen_10000.csv", index=False
    )
    atomic_npz(
        args.output / "distribucion_nula_arrays.npz",
        null_abs_r=null_abs_r,
    )
    plot_heatmaps(results, args.output)
    hashes_after = {
        path: sha256_file(Path(path)) for path in hashes_before
    }
    prior_preserved = hashes_before == hashes_after
    if not prior_preserved:
        raise RuntimeError("Cambió al menos un resultado previo.")
    write_report(
        args.output,
        results,
        summaries,
        networks,
        discrepancy,
        args.permutations,
        args.seed,
        prior_preserved,
    )
    null_result_lemon_branch = bool(
        (results["q_BH_permutation_global576"] >= 0.10).all()
        and (results["p_maxT_global576"] >= 0.05).all()
    )
    lemon_validation_actionable = bool(
        null_result_lemon_branch and len(networks) > 0
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "analysis": "simple_linear_correlations_576",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "n_subjects": N_SUBJECTS,
        "outcomes": OUTCOMES,
        "representations": REPRESENTATIONS,
        "delta_definition": "eyes_open minus eyes_closed",
        "tests_total": len(results),
        "permutations": args.permutations,
        "seed": args.seed,
        "naive_p_lt_0_05": int(
            (results["p_raw_parametric"] < 0.05).sum()
        ),
        "frozen_networks": len(networks),
        "hits_q_perm_global576_lt_0_05": int(
            (results["q_BH_permutation_global576"] < 0.05).sum()
        ),
        "hits_q_perm_global576_lt_0_10": int(
            (results["q_BH_permutation_global576"] < 0.10).sum()
        ),
        "hits_maxT_global576_lt_0_05": int(
            (results["p_maxT_global576"] < 0.05).sum()
        ),
        "null_result_triggers_lemon_branch": null_result_lemon_branch,
        "lemon_network_validation_actionable": lemon_validation_actionable,
        "lemon_network_validation_executed": False,
        "lemon_block_reason": (
            "zero_local_features_with_naive_p_lt_0_05"
            if not lemon_validation_actionable and len(networks) == 0
            else None
        ),
        "prior_hashes_before": hashes_before,
        "prior_hashes_after": hashes_after,
        "prior_results_preserved": prior_preserved,
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
            "elapsed_seconds": elapsed,
            "updated_at": datetime.now().astimezone().isoformat(),
        },
    )
    logger.info(
        "Fin: %.2f s; q576<.10=%d; maxT576<.05=%d; LEMON=%s.",
        elapsed,
        manifest["hits_q_perm_global576_lt_0_10"],
        manifest["hits_maxT_global576_lt_0_05"],
        lemon_validation_actionable,
    )
    logging.shutdown()
    write_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
