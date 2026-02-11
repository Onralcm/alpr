import base64
import requests
import streamlit as st

st.set_page_config(page_title="ALPR Turkey", page_icon="🚗", layout="wide")

# --- Minimal styling ---
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .metric-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(255,255,255,0.03);
      }
      .small { opacity: 0.75; font-size: 0.9rem; }
      .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚗 ALPR (Turkey)")
st.caption("Upload an image and get the license plate text + bounding box overlay in real-time.")

with st.sidebar:
    st.subheader("Settings")
    api_url = "http://127.0.0.1:8000/predict"
    country = st.selectbox("Country", ["TR"], index=0)
    show_debug = st.checkbox("Show debug JSON", value=False)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("Input")
    uploaded = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"])
    run = st.button("Recognize", type="primary", use_container_width=True, disabled=(uploaded is None))

    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

with colR:
    st.subheader("Result")
    if run and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        params = {"country": country}  # if you add query param later
        with st.spinner("Running detection + OCR..."):
            r = requests.post(api_url, files=files, params=params, timeout=90)

        if r.status_code != 200:
            st.error(f"API error {r.status_code}: {r.text}")
        else:
            data = r.json()

            plate = data.get("plate_text_clean") or ""
            valid = bool(data.get("is_valid_tr_format", False))
            det_conf = float(data.get("det_conf", 0.0))
            ocr_conf = float(data.get("ocr_conf", 0.0))
            timings = data.get("timings", {})

            # --- Top line ---
            pill = "✅ Valid TR format" if valid else "⚠️ Not valid TR format"
            st.markdown(f"<span class='pill'>{pill}</span>", unsafe_allow_html=True)

            st.markdown("### Plate")
            st.code(plate if plate else "(no plate read)")

            # --- Metrics row ---
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Detection conf", f"{det_conf:.2f}")
            m2.metric("OCR conf", f"{ocr_conf:.2f}")

            # Recognition-only runtime and total runtime
            t_rec = float(timings.get("t_recognition_ms", 0.0))
            t_tot = float(timings.get("t_total_ms", 0.0))
            m3.metric("Recognition (ms)", f"{t_rec:.0f}")
            m4.metric("Total (ms)", f"{t_tot:.0f}")

            tabs = st.tabs(["BBox Overlay", "Plate Crop", "Timings"])
            with tabs[0]:
                b64 = data.get("bbox_image_b64", "")
                if b64:
                    st.image(base64.b64decode(b64), use_container_width=True)
                else:
                    st.info("No overlay image returned.")

            with tabs[1]:
                crop_b64 = data.get("plate_crop_b64", "")
                if crop_b64:
                    st.image(base64.b64decode(crop_b64), caption="Cropped plate", width=420)
                else:
                    st.info("No crop returned (plate not detected).")

            with tabs[2]:
                if timings:
                    # Show key breakdown
                    show_keys = [
                        "t_decode_ms","t_detect_ms","t_crop_ms","t_ocr_ms","t_post_ms",
                        "t_draw_ms","t_encode_ms","t_recognition_ms","t_total_ms"
                    ]
                    rows = [(k, float(timings.get(k, 0.0))) for k in show_keys]
                    st.table({ "stage": [r[0] for r in rows], "ms": [f"{r[1]:.1f}" for r in rows] })
                else:
                    st.info("No timing info returned from API.")

            if show_debug:
                st.divider()
                st.json(data)
    else:
        st.info("Upload an image and click **Recognize**.")
