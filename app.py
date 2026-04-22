"""Streamlit frontend for ALPR Pro."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import pandas as pd
import streamlit as st

from src.logger import logger, setup_logger
from src.main import ALPRSystem


setup_logger("streamlit.log")

st.set_page_config(
    page_title="ALPR Pro",
    page_icon="ALPR",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_system(output_dir: Optional[str], debug: bool) -> ALPRSystem:
    """Load the backend once per Streamlit session/settings combination."""

    return ALPRSystem(output_dir=output_dir, debug=debug)


def inject_css() -> None:
    """Add lightweight custom styling for a portfolio-quality demo."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.20), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 25%),
                linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.4rem;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #164e63 58%, #0ea5e9 100%);
            color: #f8fafc;
            padding: 1.45rem 1.65rem;
            border-radius: 24px;
            box-shadow: 0 24px 55px rgba(15, 23, 42, 0.20);
            margin-bottom: 1.15rem;
        }

        .hero h1 {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2.35rem;
            letter-spacing: -0.04em;
            margin: 0;
        }

        .hero p {
            max-width: 820px;
            margin: 0.55rem 0 0 0;
            color: rgba(248, 250, 252, 0.9);
            line-height: 1.55;
        }

        .panel {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .plate-chip {
            display: inline-block;
            font-family: "Space Grotesk", sans-serif;
            font-weight: 700;
            letter-spacing: 0.04em;
            background: #e0f2fe;
            color: #0f172a;
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 0.35rem 0.65rem;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>ALPR Pro</h1>
            <p>
                Detect license plates with YOLO, enhance crops with OpenCV, recognize text with
                Tesseract or CNN OCR, and export annotated outputs for real-world demos.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_upload(uploaded_file) -> str:
    """Persist a Streamlit upload to a temporary path."""

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        return handle.name


def read_bytes(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    file_path = Path(path)
    return file_path.read_bytes() if file_path.exists() else None


def preview_image(path: str) -> None:
    image = cv2.imread(path)
    if image is None:
        st.warning("Unable to preview this image.")
        return
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)


def preview_video_first_frame(path: str) -> None:
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="First video frame", use_container_width=True)
    else:
        st.video(path)


def display_plate_card(index: int, plate: dict[str, Any], ocr_mode: str) -> None:
    """Render one detected plate with OCR and artifact previews."""

    outputs = plate.get("outputs", {})
    text = plate.get("stable_text") or plate.get("detected_text") or "OCR FAILED"
    combined = float(plate.get("combined_confidence", 0.0))
    ocr_conf = float(plate.get("ocr_confidence", 0.0))
    det_conf = float(plate.get("detection_confidence", 0.0))

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(f"**Plate {index}**")
    st.markdown(f'<span class="plate-chip">{text}</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Detection", f"{det_conf:.2f}")
    c2.metric("OCR", f"{ocr_conf:.2f}")
    c3.metric("Combined", f"{combined:.2f}")

    if not plate.get("detected_text"):
        st.warning("OCR did not return a reliable text result for this detection.")

    img_col1, img_col2 = st.columns(2)
    if outputs.get("crop"):
        img_col1.image(outputs["crop"], caption="Cropped plate", use_container_width=True)
    else:
        img_col1.info("Crop not saved.")

    if outputs.get("processed"):
        img_col2.image(outputs["processed"], caption="Processed plate", use_container_width=True)
    else:
        img_col2.info("Processed plate not saved.")

    segmented = outputs.get("segmented", [])
    if ocr_mode == "cnn":
        if segmented:
            st.image(segmented, caption=[f"Char {i + 1}" for i in range(len(segmented))], width=76)
        else:
            st.info("No segmented character images were available for this CNN result.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_downloads(result: dict[str, Any]) -> None:
    """Render download buttons for generated outputs."""

    annotated_bytes = read_bytes(result.get("annotated_path"))
    csv_bytes = read_bytes(result.get("csv_path"))
    col1, col2 = st.columns(2)

    with col1:
        if annotated_bytes:
            annotated_path = Path(result["annotated_path"])
            mime = "video/mp4" if annotated_path.suffix.lower() == ".mp4" else "image/png"
            st.download_button(
                "Download annotated output",
                data=annotated_bytes,
                file_name=annotated_path.name,
                mime=mime,
                use_container_width=True,
            )
        else:
            st.info("Annotated output is unavailable.")

    with col2:
        if csv_bytes:
            csv_path = Path(result["csv_path"])
            st.download_button(
                "Download CSV log",
                data=csv_bytes,
                file_name=csv_path.name,
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("CSV log is unavailable.")


def render_results(result: dict[str, Any], source_type: str, input_path: str, ocr_mode: str) -> None:
    """Display ALPR results."""

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Plates", result.get("plate_count", 0))
    m2.metric("OCR Mode", result.get("ocr_mode", ocr_mode).upper())
    m3.metric("Time", f"{float(result.get('processing_time', 0.0)):.2f}s")
    m4.metric("Frames", result.get("processed_frames", 1))

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Original Input")
        if source_type == "Image":
            preview_image(input_path)
        else:
            preview_video_first_frame(input_path)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Annotated Output")
        annotated_path = result.get("annotated_path")
        if annotated_path and Path(annotated_path).exists():
            if source_type == "Image":
                st.image(annotated_path, use_container_width=True)
            else:
                st.video(annotated_path)
        else:
            st.info("Annotated output was not generated.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.subheader("Recognition Results")
        results = result.get("results", [])
        if not results:
            st.warning("No license plates were detected. Try lowering confidence or using a clearer input.")
        else:
            for index, plate in enumerate(results, start=1):
                display_plate_card(index, plate, ocr_mode)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Exports")
    render_downloads(result)
    if result.get("csv_path") and Path(result["csv_path"]).exists():
        st.dataframe(pd.read_csv(result["csv_path"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_css()
    render_hero()

    with st.sidebar:
        st.header("Controls")
        source_type = st.radio("Input source", ["Image", "Video", "Webcam"], horizontal=False)
        ocr_mode = st.selectbox("OCR mode", ["tesseract", "cnn"])
        confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
        frame_skip = st.slider("Frame skip", 1, 10, 2, disabled=source_type == "Image")
        debug = st.toggle("Debug artifacts", value=False)
        webcam_index = st.number_input("Webcam index", min_value=0, max_value=10, value=0, disabled=source_type != "Webcam")

        uploaded_file = None
        if source_type == "Image":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        elif source_type == "Video":
            uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "wmv"])

        run_clicked = st.button("Run ALPR", type="primary", use_container_width=True)

    st.markdown(
        '<div class="panel">'
        "<strong>Pipeline:</strong> YOLO detection -> plate crop -> preprocessing -> OCR -> post-processing -> export."
        "</div>",
        unsafe_allow_html=True,
    )

    if not run_clicked:
        st.info("Choose an input source, configure the OCR settings, and click Run ALPR.")
        return

    try:
        system = load_system(output_dir=None, debug=debug)
    except Exception as exc:
        st.error("Backend initialization failed. Check model paths and dependency installation.")
        st.exception(exc)
        logger.exception("Streamlit backend initialization failed: {}", exc)
        return

    temp_path = None
    try:
        with st.status("Running ALPR pipeline...", expanded=True) as status:
            if source_type in {"Image", "Video"}:
                if uploaded_file is None:
                    st.warning(f"Upload a {source_type.lower()} before running inference.")
                    status.update(label="Missing input file.", state="error")
                    return
                temp_path = save_upload(uploaded_file)
                source_name = Path(uploaded_file.name).stem
            else:
                temp_path = str(int(webcam_index))
                source_name = f"webcam_{int(webcam_index)}"

            status.write("Input ready.")
            status.write("Running detector, preprocessing, OCR, and export.")

            if source_type == "Image":
                result = system.process_image(
                    image=temp_path,
                    source_name=source_name,
                    ocr_mode=ocr_mode,
                    conf_threshold=confidence,
                    save_outputs=True,
                )
            elif source_type == "Video":
                result = system.process_video(
                    video_source=temp_path,
                    source_name=source_name,
                    ocr_mode=ocr_mode,
                    conf_threshold=confidence,
                    save_outputs=True,
                    show=False,
                    frame_skip=frame_skip,
                )
            else:
                result = system.process_video(
                    video_source=int(webcam_index),
                    source_name=source_name,
                    ocr_mode=ocr_mode,
                    conf_threshold=confidence,
                    save_outputs=True,
                    show=False,
                    frame_skip=frame_skip,
                )

            status.update(label="ALPR completed.", state="complete", expanded=False)

        if result.get("plate_count", 0) == 0:
            st.warning("Inference completed, but no plates were detected.")
        elif all(not item.get("detected_text") for item in result.get("results", [])):
            st.warning("Plates were detected, but OCR did not produce reliable text.")
        else:
            st.success("ALPR inference completed successfully.")

        preview_path = temp_path if source_type != "Webcam" else result.get("annotated_path") or ""
        render_results(result, source_type if source_type != "Webcam" else "Video", preview_path, ocr_mode)

    except Exception as exc:
        st.error("ALPR processing failed.")
        st.exception(exc)
        logger.exception("Streamlit ALPR processing failed: {}", exc)


if __name__ == "__main__":
    main()
