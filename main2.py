import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import numpy as np

def main():
    # Load mô hình YOLO
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')

    model = load_model()

    # Các tham số của YOLO
    CONF_THRESHOLD = 0.3
    IMG_SIZE = 640

    # Giao diện Streamlit
    st.title("Object Detection for Images and Videos")
    st.write("Upload an image or video to detect objects using YOLOv11.")

    # Tải file ảnh hoặc video
    file = st.file_uploader("Upload Image or Video", type=['jpg', 'png', 'jpeg', 'mp4', 'avi'])
    if file is not None:
        file_type = file.name.split('.')[-1].lower()
        if file_type in ['jpg', 'png', 'jpeg']:
            # Xử lý file ảnh
            st.image(file, caption="Uploaded Image", use_container_width=True)
            image = Image.open(file).convert('RGB')  # Đảm bảo ảnh ở định dạng RGB

            # Chạy mô hình
            result = model.predict(
                source=image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD
            )

            # Annotate kết quả
            annotated_img = result[0].plot()

            # Chuyển đổi BGR (OpenCV) sang RGB (PIL)
            annotated_img_rgb = annotated_img[..., ::-1]  # Chuyển BGR sang RGB
            annotated_img_pil = Image.fromarray(annotated_img_rgb)

            # Hiển thị kết quả đã annotate
            st.image(annotated_img_pil, caption="Detected Objects", use_container_width=True)

        elif file_type in ['mp4', 'avi']:
            # Xử lý file video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            video_path = tfile.name

            cap = cv2.VideoCapture(video_path)

            # Đường dẫn tạm để lưu video kết quả
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Xử lý từng frame
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Chạy YOLO trên từng frame
                results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
                annotated_frame = results[0].plot()

                # Ghi frame đã annotate vào video kết quả
                out.write(annotated_frame)

                # Hiển thị frame tạm thời
                stframe.image(annotated_frame[..., ::-1], channels="RGB", use_container_width=True)

            cap.release()
            out.release()

            # Hiển thị video kết quả
            st.video(output_path)

if __name__ == "__main__":
    main()