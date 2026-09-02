import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Object Detection AI", layout="wide")
st.title("Object Detection AI")
st.write("Upload an image and detect objects using YOLOv8")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        model = YOLO("yolov8n.pt")
        results = model(image)

        result = results[0]
        result_img = result.plot()
        st.image(result_img, caption="Detected Image", use_container_width=True)

        # Object list + confidence
        names = model.names
        boxes = result.boxes
        data = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0]) * 100
                data.append([label, f"{conf:.2f}%"])

            df = pd.DataFrame(data, columns=["Object", "Confidence"])
            st.subheader("Detected Objects")
            st.dataframe(df, use_container_width=True)

            # Download detected image
            result_pil = Image.fromarray(result_img)
            buffer = BytesIO()
            result_pil.save(buffer, format="PNG")
            byte_data = buffer.getvalue()

            st.download_button(
                label="Download Detected Image",
                data=byte_data,
                file_name="detected_image.png",
                mime="image/png"
            )

            st.success(f"Detection completed. {len(data)} objects found.")
        else:
            st.warning("No objects detected.")