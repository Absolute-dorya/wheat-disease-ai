"""
Wheat Disease Detection - Streamlit web app.

Tabs:
  1. Diagnose   - upload one or many leaf photos -> class + confidence + Grad-CAM + advisory
  2. Map & Alerts - geotagged reports, hotspot clustering, active advisories

Run:
    streamlit run app/app.py
"""
import io
from datetime import datetime, timezone
from pathlib import Path

import folium
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from folium.plugins import MarkerCluster
from PIL import Image
from streamlit_folium import st_folium
from torchvision import transforms

st.set_page_config(page_title="Wheat Disease Detection", page_icon="🌾", layout="wide")

CKPT_CANDIDATES = ["models/best.pt", "results/best.pt"] + [
    str(p) for p in sorted(Path("results").glob("*/best.pt"))
]

# --------------------------------------------------------------------------
# Advisory rules. Keep these EXPLICIT and rule-based so they are explainable
# in the paper. Replace with locally validated extension guidance before any
# real-world deployment.
# --------------------------------------------------------------------------
ADVISORY = {
    "healthy":        ("No action", "Leaf appears healthy. Continue routine monitoring."),
    "leaf_rust":      ("High priority", "Isolate and report. Apply a recommended triazole/strobilurin "
                                        "fungicide per local extension guidance. Scout neighbouring rows."),
    "stripe_rust":    ("URGENT", "Yellow/stripe rust spreads rapidly by wind. Report immediately and "
                                 "treat the field and buffer zone. Alert neighbouring farms."),
    "powdery_mildew": ("Medium priority", "Improve airflow, avoid excess nitrogen. Apply sulphur or "
                                          "a labelled fungicide if coverage exceeds ~5% of leaf area."),
    "septoria":       ("Medium priority", "Remove crop residue after harvest; apply fungicide at "
                                          "early symptom onset. Rotate away from wheat next season."),
    "fusarium_head_blight": ("High priority", "Risk of mycotoxin contamination. Do not use grain for "
                                              "feed without testing. Apply fungicide at flowering."),
    "loose_smut":     ("High priority", "Seed-borne. Discard seed stock and use certified treated seed "
                                        "next season."),
}
DEFAULT_ADVISORY = ("Review", "Consult your local agricultural extension officer.")


def advisory_for(label: str, confidence: float):
    title, body = ADVISORY.get(label.lower().replace(" ", "_"), DEFAULT_ADVISORY)
    if title in ("URGENT", "High priority") and confidence < 0.70:
        return "Review (low confidence)", (
            f"Model suggests {label} but confidence is only {confidence:.0%}. "
            f"Re-photograph in even daylight against a plain background, then confirm with an "
            f"extension officer. Indicative guidance: {body}"
        )
    return title, body


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def find_ckpt():
    for c in CKPT_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    import timm
    model = timm.create_model(ckpt["backbone"], pretrained=False,
                              num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"], int(ckpt.get("img_size", 224))


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# --------------------------------------------------------------------------
# Grad-CAM (minimal, no extra dependency)
# --------------------------------------------------------------------------
class GradCAM:
    """Hooks the last convolutional-ish block. Works for timm CNN backbones."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        self._handles = []

    def _target_layer(self):
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        raise RuntimeError("No Conv2d layer found - use --backbone with a CNN for Grad-CAM.")

    def __enter__(self):
        layer = self._target_layer()
        self._handles.append(layer.register_forward_hook(self._save_act))
        self._handles.append(layer.register_full_backward_hook(self._save_grad))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        return False

    def _save_act(self, _m, _i, out):
        self.acts = out.detach()

    def _save_grad(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP over grads
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True)) # weighted sum + ReLU
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy(), class_idx, F.softmax(logits, dim=1)[0].detach().numpy()


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-style heatmap over the image without matplotlib."""
    import matplotlib.cm as cm
    base = np.asarray(img.resize((cam.shape[1], cam.shape[0]))).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def lesion_ratio(img: Image.Image):
    """Heuristic severity: fraction of pixels far from healthy green in HSV.

    This is a documented HEURISTIC, not a validated severity scale. Say so in
    the paper. Replace with a trained severity model before deployment.
    """
    small = np.asarray(img.convert("RGB").resize((128, 128))).astype(np.float32) / 255.0
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    mx, mn = small.max(2), small.min(2)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-6, None), 0)
    hue = np.zeros_like(mx)
    mask = mx > 0
    d = np.clip(mx - mn, 1e-6, None)
    hue = np.where(mask & (mx == r), ((g - b) / d) % 6, hue)
    hue = np.where(mask & (mx == g), (b - r) / d + 2, hue)
    hue = np.where(mask & (mx == b), (r - g) / d + 4, hue)
    hue = hue * 60.0 / 360.0
    is_green = (hue > 0.20) & (hue < 0.48) & (s > 0.15) & (v > 0.12)
    return float(1.0 - is_green.mean())


def severity_bucket(ratio: float):
    if ratio < 0.15:
        return "Healthy / negligible", 0
    if ratio < 0.35:
        return "Mild", 1
    if ratio < 0.60:
        return "Moderate", 2
    return "Severe", 3


# --------------------------------------------------------------------------
# Session state for the report map
# --------------------------------------------------------------------------
if "reports" not in st.session_state:
    st.session_state.reports = []

st.title("🌾 Wheat Disease Detection")
st.caption("Upload a wheat leaf photo for instant AI diagnosis, severity grading and an advisory. "
           "Reports appear on the map so hotspots can be tracked.")

ckpt_path = find_ckpt()
if ckpt_path is None:
    st.error("No trained model found. Train one first:\n\n"
             "```\npython src/train.py --data data/processed --backbone efficientnet_b0 --epochs 15\n```\n\n"
             "Then re-run this app. Expected at `results/<run>/best.pt`.")
    st.stop()

model, classes, img_size = load_model(ckpt_path)
tf = make_transform(img_size)
st.sidebar.success(f"Model: `{Path(ckpt_path).name}`\n\nClasses: {', '.join(classes)}")

tab_predict, tab_map = st.tabs(["🔍 Diagnose", "🗺️ Map & Alerts"])

# --------------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1])
    with left:
        files = st.file_uploader("Wheat leaf photos (JPG/PNG)",
                                 type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True)
        show_cam = st.checkbox("Show Grad-CAM explanation", value=True)
        col_a, col_b = st.columns(2)
        lat = col_a.number_input("Latitude (optional)", value=0.0, format="%.5f")
        lon = col_b.number_input("Longitude (optional)", value=0.0, format="%.5f")

    if files:
        for f in files:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            x = tf(img).unsqueeze(0)

            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0].numpy()
            idx = int(probs.argmax())
            label, conf = classes[idx], float(probs[idx])

            cam_img = None
            if show_cam:
                try:
                    with GradCAM(model) as gcam:
                        cam, _, _ = gcam(x, class_idx=idx)
                    cam_img = overlay_cam(img, cam)
                except Exception as e:
                    st.caption(f"Grad-CAM unavailable: {e}")

            with right:
                st.subheader(f"{f.name}")
                c1, c2 = st.columns(2)
                c1.image(img, caption="Uploaded", use_container_width=True)
                if cam_img is not None:
                    c2.image(cam_img, caption="Model attention (Grad-CAM)",
                             use_container_width=True)

                if conf >= 0.70:
                    st.success(f"**{label.replace('_', ' ').title()}** — confidence {conf:.1%}")
                else:
                    st.warning(f"**{label.replace('_', ' ').title()}** — confidence only {conf:.1%}. "
                               f"Re-photograph in even daylight before trusting this.")

                st.bar_chart({classes[i].replace("_", " "): float(probs[i])
                              for i in range(len(classes))})

                ratio = lesion_ratio(img)
                sev, sev_level = severity_bucket(ratio)
                st.metric("Estimated affected leaf area", f"{ratio:.1%}", sev)
                if sev_level >= 2:
                    st.error(f"Severity: **{sev}** — escalate")
                st.caption("Affected-area estimate is a colour-based heuristic, not a validated "
                           "severity scale.")

                title, body = advisory_for(label, conf)
                st.info(f"**Advisory — {title}**\n\n{body}")

            if lat and lon:
                st.session_state.reports.append({
                    "lat": float(lat), "lon": float(lon), "label": label,
                    "confidence": conf, "severity": sev,
                    "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                })
            st.divider()

# --------------------------------------------------------------------------
with tab_map:
    st.subheader("Reported cases & disease hotspots")
    reports = st.session_state.reports

    if not reports:
        st.info("No geotagged reports yet. Add a latitude/longitude in the Diagnose tab "
                "and classify an image to place a pin.")
    else:
        if st.button("Clear all reports"):
            st.session_state.reports = []
            st.rerun()

        m = folium.Map(location=[reports[-1]["lat"], reports[-1]["lon"]],
                       zoom_start=11, tiles="OpenStreetMap")
        cluster = MarkerCluster().add_to(m)
        colors = {"healthy": "green", "leaf_rust": "orange", "stripe_rust": "red",
                  "powdery_mildew": "blue", "septoria": "purple",
                  "fusarium_head_blight": "darkred", "loose_smut": "darkpurple"}

        for r in reports:
            folium.Marker(
                [r["lat"], r["lon"]],
                popup=(f"<b>{r['label'].replace('_', ' ').title()}</b><br>"
                       f"Confidence: {r['confidence']:.0%}<br>"
                       f"Severity: {r['severity']}<br>{r['time']}"),
                icon=folium.Icon(color=colors.get(r["label"], "gray"), icon="leaf",
                                 prefix="fa"),
            ).add_to(cluster)

        st_folium(m, height=460, use_container_width=True)

        counts = {}
        for r in reports:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        st.write("**Case counts**")
        st.bar_chart(counts)

        urgent = [r for r in reports
                  if r["label"] in ("stripe_rust", "leaf_rust", "fusarium_head_blight",
                                    "loose_smut") and r["confidence"] >= 0.70]
        if urgent:
            st.error(f"⚠️ **{len(urgent)} high-priority case(s)** — rapid-spreading rust/blight "
                     f"detected. Recommend immediate scouting and treatment of the affected area.")
        else:
            st.success("No high-priority cases above the confidence threshold.")
