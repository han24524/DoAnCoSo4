import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
from moviepy.editor import VideoFileClip  # Import MoviePy


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
    st.title("Helmet Detection for Images and Videos")
    st.write("Upload an image or video to detect objects using YOLOv11.")

    # Tải file ảnh hoặc video
    file = st.file_uploader("Upload Image or Video", type=[
                            'jpg', 'png', 'jpeg', 'mp4', 'avi'])
    if file is not None:
        file_type = file.name.split('.')[-1].lower()
        if file_type in ['jpg', 'png', 'jpeg']:
            # Xử lý file ảnh
            st.image(file, caption="Uploaded Image", use_container_width=True)
            image = Image.open(file).convert(
                'RGB')  # Đảm bảo ảnh ở định dạng RGB

            # Chạy mô hình
            result = model.predict(
                source=image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)

            # Annotate kết quả
            annotated_img = result[0].plot()

            # Chuyển đổi BGR (OpenCV) sang RGB (PIL)
            st.image(
                annotated_img[..., ::-1], caption="Detected Objects", use_container_width=True)

        elif file_type in ['mp4', 'avi']:
            # Xử lý file video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(file.read())
            video_path = tfile.name

            # Đường dẫn tạm để lưu video kết quả
            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4').name

            # Hiển thị thông báo xử lý
            st.write("Processing video. Please wait...")

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Xử lý từng frame và ghi vào video đầu ra
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break

                # Chạy YOLO trên từng frame
                results = model.predict(
                    frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                # Cập nhật progress bar
                progress_bar.progress((i + 1) / frame_count)

            cap.release()
            out.release()

            st.success("Video processing completed!")

            # Sử dụng MoviePy để nén video
            compressed_output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4').name
            st.write("Compressing video using MoviePy...")
            video_clip = VideoFileClip(output_path)
            video_clip.write_videofile(
                compressed_output_path, codec='libx264', audio=False)
            video_clip.close()

            # Hiển thị video kết quả
            st.video(compressed_output_path)


if __name__ == "__main__":
    main()
