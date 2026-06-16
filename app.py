from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image

# ============================================================
# Конфигурация приложения
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEGMENTATION_CKPT = ARTIFACTS_DIR / "best_damage_segmentation_binary.pt"
THRESHOLD_SWEEP_PATH = ARTIFACTS_DIR / "binary_threshold_sweep.csv"
HYBRID_WEAR_PATH = ARTIFACTS_DIR / "hybrid_wear_regressor.joblib"
HYBRID_RISK_PATH = ARTIFACTS_DIR / "hybrid_poor_within_horizon_clf.joblib"
HYBRID_RUL_PATH = ARTIFACTS_DIR / "hybrid_rul_regressor.joblib"
HYBRID_DEFAULTS_PATH = ARTIFACTS_DIR / "hybrid_feature_defaults.json"
HYBRID_METRICS_PATH = ARTIFACTS_DIR / "hybrid_metrics.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
BACKBONE = "resnet50"
SEGMENTATION_ARCH = "deeplabv3plus"
MASK_PROB_DEFAULT_THRESHOLD = 0.45
MIN_COMPONENT_AREA_PX = 48
USE_TTA_INFERENCE = True

TABULAR_FEATURE_COLS = [
    "age", "adt", "length", "main_spans",
    "deck_condition", "super_condition", "sub_condition",
    "min_condition", "mean_condition",
    "delta_deck", "delta_super", "delta_sub",
]
VISUAL_FEATURE_COLS = [
    "visual_severity",
    "damage_area_ratio_union",
    "damage_perimeter_ratio",
    "largest_component_ratio",
    "components_count",
    "component_density_per_100kpx",
    "mean_width_px",
    "max_width_px",
    "length_proxy_px",
    "fragmentation_index",
    "elongation_proxy",
]
HYBRID_FEATURE_COLS = TABULAR_FEATURE_COLS + VISUAL_FEATURE_COLS


# ============================================================
# Модель и инференс
# ============================================================

def build_segmentation_model() -> nn.Module:
    if SEGMENTATION_ARCH.lower() == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=BACKBONE,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    elif SEGMENTATION_ARCH.lower() == "fpn":
        model = smp.FPN(
            encoder_name=BACKBONE,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    else:
        model = smp.UnetPlusPlus(
            encoder_name=BACKBONE,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
    return model


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Не удалось извлечь state_dict из checkpoint.")


def read_rgb_from_any(image_input) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8)
    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"), dtype=np.uint8)
    return np.array(Image.open(image_input).convert("RGB"), dtype=np.uint8)


def preprocess_image(image: np.ndarray, image_size: int = IMAGE_SIZE) -> Tuple[np.ndarray, torch.Tensor]:
    image_resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    return image_resized, x


def postprocess_binary_mask(prob_map: np.ndarray, threshold: float = 0.5, min_area_px: int = MIN_COMPONENT_AREA_PX) -> np.ndarray:
    mask = (prob_map >= threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    dynamic_min_area = max(min_area_px, int(0.00015 * mask.size))
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= dynamic_min_area:
            cleaned[labels == lab] = 1

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned.astype(np.uint8)


def normalize_binary_damage_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def extract_damage_features_from_mask(mask: np.ndarray) -> Dict[str, float]:
    mask = normalize_binary_damage_mask(mask)
    h, w = mask.shape
    total_pixels = float(h * w)
    area_px = float(mask.sum())
    area_ratio = area_px / total_pixels if total_pixels else 0.0

    if area_px <= 0:
        return {
            "damage_area_px": 0.0,
            "damage_area_ratio_union": 0.0,
            "damage_perimeter_px": 0.0,
            "damage_perimeter_ratio": 0.0,
            "largest_component_px": 0.0,
            "largest_component_ratio": 0.0,
            "components_count": 0.0,
            "component_density_per_100kpx": 0.0,
            "mean_width_px": 0.0,
            "max_width_px": 0.0,
            "length_proxy_px": 0.0,
            "fragmentation_index": 0.0,
            "elongation_proxy": 0.0,
        }

    mask_u8 = (mask * 255).astype(np.uint8)
    edges = cv2.Canny(mask_u8, 50, 150)
    perimeter_px = float((edges > 0).sum())
    perimeter_ratio = perimeter_px / total_pixels if total_pixels else 0.0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    components_count = float(len(component_areas))
    largest_component_px = float(max(component_areas)) if component_areas else 0.0
    largest_component_ratio = largest_component_px / total_pixels if total_pixels else 0.0
    component_density = components_count / max(total_pixels / 100000.0, 1e-6)

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    positive_dist = dist[mask > 0]
    mean_half_width = float(positive_dist.mean()) if positive_dist.size else 0.0
    max_half_width = float(positive_dist.max()) if positive_dist.size else 0.0
    mean_width_px = 2.0 * mean_half_width
    max_width_px = 2.0 * max_half_width

    length_proxy_px = area_px / max(mean_width_px, 1.0)
    fragmentation_index = 1.0 - (largest_component_px / max(area_px, 1.0))
    elongation_proxy = perimeter_px / max(np.sqrt(area_px), 1.0)

    return {
        "damage_area_px": area_px,
        "damage_area_ratio_union": area_ratio,
        "damage_perimeter_px": perimeter_px,
        "damage_perimeter_ratio": perimeter_ratio,
        "largest_component_px": largest_component_px,
        "largest_component_ratio": largest_component_ratio,
        "components_count": components_count,
        "component_density_per_100kpx": component_density,
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "length_proxy_px": float(length_proxy_px),
        "fragmentation_index": float(np.clip(fragmentation_index, 0.0, 1.0)),
        "elongation_proxy": float(elongation_proxy),
    }


def compute_visual_severity_from_mask(mask: np.ndarray) -> Dict[str, float]:
    feats = extract_damage_features_from_mask(mask)
    area_term = min(feats["damage_area_ratio_union"] / 0.050, 1.0)
    perimeter_term = min(feats["damage_perimeter_ratio"] / 0.100, 1.0)
    width_term = min(feats["mean_width_px"] / 10.0, 1.0)
    length_term = min(feats["length_proxy_px"] / 220.0, 1.0)
    fragment_term = min(feats["fragmentation_index"] / 0.65, 1.0)

    severity_score = (
        0.34 * area_term
        + 0.18 * perimeter_term
        + 0.18 * width_term
        + 0.20 * length_term
        + 0.10 * fragment_term
    )
    severity_score = float(np.clip(severity_score, 0.0, 1.0))

    if severity_score < 0.18:
        grade = "low"
    elif severity_score < 0.40:
        grade = "moderate"
    elif severity_score < 0.68:
        grade = "high"
    else:
        grade = "critical"

    return {**feats, "visual_severity": severity_score, "visual_grade": grade}


def estimate_photo_only_wear(visual_features: Dict[str, float]) -> Dict[str, float]:
    area_term = float(np.clip(visual_features.get("damage_area_ratio_union", 0.0) / 0.050, 0.0, 1.0))
    width_term = float(np.clip(visual_features.get("mean_width_px", 0.0) / 10.0, 0.0, 1.0))
    length_term = float(np.clip(visual_features.get("length_proxy_px", 0.0) / 220.0, 0.0, 1.0))
    fragment_term = float(np.clip(visual_features.get("fragmentation_index", 0.0) / 0.65, 0.0, 1.0))
    visual_severity = float(np.clip(visual_features.get("visual_severity", 0.0), 0.0, 1.0))

    photo_wear_index = float(np.clip(
        0.55 * visual_severity + 0.15 * area_term + 0.12 * width_term + 0.10 * length_term + 0.08 * fragment_term,
        0.0,
        1.0,
    ))

    if photo_wear_index < 0.18:
        photo_grade = "low"
    elif photo_wear_index < 0.40:
        photo_grade = "moderate"
    elif photo_wear_index < 0.68:
        photo_grade = "high"
    else:
        photo_grade = "critical"

    return {
        "photo_wear_index": photo_wear_index,
        "photo_wear_grade": photo_grade,
    }


def russian_grade(grade: str) -> str:
    return {
        "low": "Низкая",
        "moderate": "Умеренная",
        "high": "Высокая",
        "critical": "Критическая",
    }.get(grade, grade)


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image_rgb.copy().astype(np.uint8)
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = image.copy()
    overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([255, 50, 40])).astype(np.uint8)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    return overlay


@dataclass
class LoadedModels:
    segmentation_model: Optional[nn.Module]
    wear_model: Optional[object]
    risk_model: Optional[object]
    rul_model: Optional[object]
    hybrid_feature_defaults: Dict[str, float]
    threshold: float


class Predictor:
    def __init__(self):
        self.models = LoadedModels(None, None, None, None, {}, MASK_PROB_DEFAULT_THRESHOLD)

    def load_from_artifacts_dir(self, artifacts_dir: Path) -> str:
        artifacts_dir = Path(artifacts_dir)
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Папка артефактов не найдена: {artifacts_dir}")

        copied = []
        for src in artifacts_dir.glob("*"):
            dst = ARTIFACTS_DIR / src.name
            if src.is_file():
                shutil.copy2(src, dst)
                copied.append(src.name)

        self._load_all()
        return "Загружены артефакты: " + ", ".join(sorted(copied)) if copied else "Артефакты не найдены."

    def load_from_zip(self, zip_path: str) -> str:
        if not zip_path:
            raise ValueError("Не передан zip-файл с артефактами.")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ARTIFACTS_DIR)
        self._load_all()
        return f"Артефакты распакованы в {ARTIFACTS_DIR}"

    def _load_all(self):
        if SEGMENTATION_CKPT.exists():
            model = build_segmentation_model().to(DEVICE)
            ckpt = torch.load(SEGMENTATION_CKPT, map_location=DEVICE)
            state = _extract_state_dict(ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()
            self.models.segmentation_model = model
        else:
            self.models.segmentation_model = None

        self.models.wear_model = joblib.load(HYBRID_WEAR_PATH) if HYBRID_WEAR_PATH.exists() else None
        self.models.risk_model = joblib.load(HYBRID_RISK_PATH) if HYBRID_RISK_PATH.exists() else None
        self.models.rul_model = joblib.load(HYBRID_RUL_PATH) if HYBRID_RUL_PATH.exists() else None

        if HYBRID_DEFAULTS_PATH.exists():
            self.models.hybrid_feature_defaults = json.loads(HYBRID_DEFAULTS_PATH.read_text(encoding="utf-8"))
        else:
            self.models.hybrid_feature_defaults = {c: 0.0 for c in HYBRID_FEATURE_COLS}

        threshold = MASK_PROB_DEFAULT_THRESHOLD
        if THRESHOLD_SWEEP_PATH.exists():
            try:
                sweep_df = pd.read_csv(THRESHOLD_SWEEP_PATH)
                if len(sweep_df) and "threshold" in sweep_df.columns:
                    if "dice" in sweep_df.columns:
                        sweep_df = sweep_df.sort_values(["dice", "iou"], ascending=False)
                    threshold = float(sweep_df.iloc[0]["threshold"])
            except Exception:
                threshold = MASK_PROB_DEFAULT_THRESHOLD
        self.models.threshold = threshold

    def status(self) -> Dict[str, object]:
        return {
            "segmentation_loaded": self.models.segmentation_model is not None,
            "wear_loaded": self.models.wear_model is not None,
            "risk_loaded": self.models.risk_model is not None,
            "rul_loaded": self.models.rul_model is not None,
            "threshold": self.models.threshold,
            "artifacts_dir": str(ARTIFACTS_DIR),
        }

    @torch.no_grad()
    def predict(self, image_input, bridge_features: Optional[Dict[str, float]] = None, threshold: Optional[float] = None):
        if self.models.segmentation_model is None:
            raise RuntimeError(
                "Сегментационная модель не загружена. Поместите best_damage_segmentation_binary.pt в папку artifacts "
                "или загрузите zip с артефактами через интерфейс."
            )
        image = read_rgb_from_any(image_input)
        image_resized, x = preprocess_image(image)
        x = x.to(DEVICE)
        logits = self.models.segmentation_model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        if USE_TTA_INFERENCE:
            x_flip = torch.flip(x, dims=[3])
            logits_flip = self.models.segmentation_model(x_flip)
            prob_flip = torch.sigmoid(logits_flip)[0, 0].detach().cpu().numpy()
            prob = 0.5 * prob + 0.5 * np.fliplr(prob_flip)

        thr = self.models.threshold if threshold is None else float(threshold)
        pred_mask = postprocess_binary_mask(prob, threshold=thr)
        overlay = overlay_mask_on_image(image_resized, pred_mask)
        visual = compute_visual_severity_from_mask(pred_mask)
        photo = estimate_photo_only_wear(visual)
        hybrid = self.predict_hybrid(visual, bridge_features)

        result = {
            **visual,
            **photo,
            **hybrid,
            "threshold_used": float(thr),
        }
        return image_resized, overlay, pred_mask * 255, result

    def predict_hybrid(self, visual_features: Dict[str, float], bridge_features: Optional[Dict[str, float]]) -> Dict[str, object]:
        if bridge_features is None:
            bridge_features = {}
        defaults = dict(self.models.hybrid_feature_defaults or {c: 0.0 for c in HYBRID_FEATURE_COLS})
        row = {**defaults, **visual_features}

        if bridge_features:
            row["age"] = _safe_num(bridge_features.get("age"), defaults.get("age", 0.0))
            row["adt"] = _safe_num(bridge_features.get("adt"), defaults.get("adt", 0.0))
            row["length"] = _safe_num(bridge_features.get("length"), defaults.get("length", 0.0))
            row["main_spans"] = _safe_num(bridge_features.get("main_spans"), defaults.get("main_spans", 0.0))
            row["deck_condition"] = _safe_num(bridge_features.get("deck_condition"), defaults.get("deck_condition", 0.0))
            row["super_condition"] = _safe_num(bridge_features.get("super_condition"), defaults.get("super_condition", 0.0))
            row["sub_condition"] = _safe_num(bridge_features.get("sub_condition"), defaults.get("sub_condition", 0.0))
            row["delta_deck"] = _safe_num(bridge_features.get("delta_deck"), defaults.get("delta_deck", 0.0))
            row["delta_super"] = _safe_num(bridge_features.get("delta_super"), defaults.get("delta_super", 0.0))
            row["delta_sub"] = _safe_num(bridge_features.get("delta_sub"), defaults.get("delta_sub", 0.0))
            row["min_condition"] = min(row["deck_condition"], row["super_condition"], row["sub_condition"])
            row["mean_condition"] = float(np.mean([row["deck_condition"], row["super_condition"], row["sub_condition"]]))
        else:
            # photo-only режим
            row["min_condition"] = defaults.get("min_condition", 0.0)
            row["mean_condition"] = defaults.get("mean_condition", 0.0)

        X = pd.DataFrame([{k: row.get(k, 0.0) for k in HYBRID_FEATURE_COLS}])
        out = {}
        if self.models.wear_model is not None:
            wear_idx = float(self.models.wear_model.predict(X)[0])
            out["hybrid_wear_index"] = float(np.clip(wear_idx, 0.0, 1.0))
            out["hybrid_wear_grade"] = _grade_from_wear_index(out["hybrid_wear_index"])
        else:
            out["hybrid_wear_index"] = None
            out["hybrid_wear_grade"] = None

        if self.models.risk_model is not None:
            if hasattr(self.models.risk_model, "predict_proba"):
                out["poor_within_horizon_prob"] = float(self.models.risk_model.predict_proba(X)[0, 1])
            else:
                out["poor_within_horizon_prob"] = float(self.models.risk_model.predict(X)[0])
        else:
            out["poor_within_horizon_prob"] = None

        if self.models.rul_model is not None:
            out["years_to_poor_pred"] = max(0.0, float(self.models.rul_model.predict(X)[0]))
        else:
            out["years_to_poor_pred"] = None
        return out


def _grade_from_wear_index(x: float) -> str:
    x = float(np.clip(x, 0.0, 1.0))
    if x < 0.18:
        return "low"
    elif x < 0.40:
        return "moderate"
    elif x < 0.68:
        return "high"
    return "critical"


def _safe_num(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


PREDICTOR = Predictor()
if any(p.exists() for p in [SEGMENTATION_CKPT, HYBRID_WEAR_PATH, HYBRID_RISK_PATH, HYBRID_RUL_PATH]):
    try:
        PREDICTOR._load_all()
    except Exception as e:
        print("Auto-load artifacts failed:", e)


# ============================================================
# Интерфейс
# ============================================================

def load_zip_artifacts(zip_file):
    if zip_file is None:
        return PREDICTOR.status(), "Файл архива не выбран."
    try:
        msg = PREDICTOR.load_from_zip(zip_file)
        return PREDICTOR.status(), msg
    except Exception as e:
        return PREDICTOR.status(), f"Ошибка загрузки артефактов: {e}"


def analyze_image(
    image,
    photo_only_mode,
    age,
    adt,
    length,
    main_spans,
    deck_condition,
    super_condition,
    sub_condition,
    delta_deck,
    delta_super,
    delta_sub,
    threshold,
):
    if image is None:
        raise gr.Error("Сначала загрузите фотографию.")

    bridge_features = None
    if not photo_only_mode:
        bridge_features = {
            "age": age,
            "adt": adt,
            "length": length,
            "main_spans": main_spans,
            "deck_condition": deck_condition,
            "super_condition": super_condition,
            "sub_condition": sub_condition,
            "delta_deck": delta_deck,
            "delta_super": delta_super,
            "delta_sub": delta_sub,
        }

    try:
        src, overlay, mask, result = PREDICTOR.predict(image, bridge_features=bridge_features, threshold=threshold)
    except Exception as e:
        raise gr.Error(str(e))

    df = pd.DataFrame(
        [
            ["Визуальная степень износа", f"{result['photo_wear_index'] * 100:.1f}%", russian_grade(result["photo_wear_grade"])],
            ["Площадь повреждения", f"{result['damage_area_ratio_union'] * 100:.2f}%", "по маске"],
            ["Средняя ширина дефекта", f"{result['mean_width_px']:.2f} px", "proxy"],
            ["Максимальная ширина дефекта", f"{result['max_width_px']:.2f} px", "proxy"],
            ["Длина дефекта", f"{result['length_proxy_px']:.2f} px", "proxy"],
            ["Фрагментация", f"{result['fragmentation_index']:.3f}", "индекс"],
            ["Порог маски", f"{result['threshold_used']:.2f}", "использовано"],
            ["Гибридный wear", (f"{result['hybrid_wear_index']*100:.1f}%" if result['hybrid_wear_index'] is not None else "н/д"), russian_grade(result['hybrid_wear_grade']) if result['hybrid_wear_grade'] else "н/д"],
            ["Риск деградации", (f"{result['poor_within_horizon_prob']*100:.1f}%" if result['poor_within_horizon_prob'] is not None else "н/д"), "вероятность"],
            ["RUL-proxy", (f"{result['years_to_poor_pred']:.1f} лет" if result['years_to_poor_pred'] is not None else "н/д"), "прогноз"],
        ],
        columns=["Показатель", "Значение", "Комментарий"],
    )

    fig = plt.figure(figsize=(7, 4))
    items = [
        ("Площадь", result["damage_area_ratio_union"] * 100),
        ("Ширина", min(result["mean_width_px"] * 10, 100)),
        ("Длина", min(result["length_proxy_px"] / 3, 100)),
        ("Износ", result["photo_wear_index"] * 100),
    ]
    names = [x[0] for x in items]
    values = [x[1] for x in items]
    plt.bar(names, values)
    plt.ylim(0, 100)
    plt.ylabel("Нормированная оценка")
    plt.title("Профиль визуального износа по изображению")
    plt.tight_layout()

    summary = (
        f"Визуальная оценка износа: {result['photo_wear_index'] * 100:.1f}% "
        f"({russian_grade(result['photo_wear_grade']).lower()}). "
        f"Доля повреждённой области составляет {result['damage_area_ratio_union'] * 100:.2f}% от анализируемого кадра."
    )
    if result.get("poor_within_horizon_prob") is not None:
        summary += f" Вероятность ухудшения состояния в горизонте прогноза: {result['poor_within_horizon_prob'] * 100:.1f}%."
    if result.get("years_to_poor_pred") is not None:
        summary += f" Оценка RUL-proxy: {result['years_to_poor_pred']:.1f} лет."

    return src, overlay, mask, df, fig, summary


def build_demo() -> gr.Blocks:
    status_value = json.dumps(PREDICTOR.status(), ensure_ascii=False, indent=2)
    with gr.Blocks(title="Оценка степени износа инфраструктурных объектов") as demo:
        gr.Markdown("""
# Оценка степени износа инфраструктурных объектов по фотографии

""")
        with gr.Tab("Анализ изображения"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Фотография дефекта")
                    photo_only = gr.Checkbox(value=True, label="Оценивать только по фотографии")
                    threshold = gr.Slider(0.2, 0.8, value=PREDICTOR.models.threshold, step=0.01, label="Порог бинарной маски")
                    gr.Markdown("### Параметры объекта (используются только при снятой галочке выше)")
                    age = gr.Number(value=42, label="Возраст объекта, лет")
                    adt = gr.Number(value=18000, label="Интенсивность движения (ADT)")
                    length = gr.Number(value=120, label="Длина сооружения, м")
                    main_spans = gr.Number(value=3, label="Количество пролётов")
                    deck_condition = gr.Number(value=6, label="Состояние проезжей части")
                    super_condition = gr.Number(value=6, label="Состояние пролётного строения")
                    sub_condition = gr.Number(value=5, label="Состояние опор")
                    delta_deck = gr.Number(value=-1, label="Изменение состояния проезжей части")
                    delta_super = gr.Number(value=0, label="Изменение состояния пролётного строения")
                    delta_sub = gr.Number(value=0, label="Изменение состояния опор")
                    run_btn = gr.Button("Рассчитать состояние", variant="primary")
                with gr.Column(scale=1):
                    src_out = gr.Image(label="Исходное изображение")
                    overlay_out = gr.Image(label="Карта повреждений")
                    mask_out = gr.Image(label="Бинарная маска")
            with gr.Row():
                table_out = gr.Dataframe(label="Результаты")
            with gr.Row():
                plot_out = gr.Plot(label="График визуального износа")
            summary_out = gr.Markdown()

            run_btn.click(
                analyze_image,
                inputs=[image_input, photo_only, age, adt, length, main_spans, deck_condition, super_condition, sub_condition, delta_deck, delta_super, delta_sub, threshold],
                outputs=[src_out, overlay_out, mask_out, table_out, plot_out, summary_out],
            )

       
            
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
