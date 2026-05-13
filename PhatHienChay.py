import cv2
import numpy as np
import gradio as gr
import os
import moviepy.editor as mp
from pydub.generators import Sine
from PIL import Image, ImageDraw, ImageFont


# ================================================================================================================================
# CÁC HÀM HỖ TRỢ


# Hàm tạo âm thanh
def ensure_beep_file():
    if not os.path.exists("beep.wav"):
        beep = Sine(1000).to_audio_segment(duration=500)
        beep.export("beep.wav", format="wav")


ensure_beep_file()


# Hàm vẽ chữ Tiếng Việt
def draw_vietnamese_text(img, text, pos, color, font_size=18):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Dùng font Arial có sẵn
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    # Font dự phòng
    except IOError:
        font = ImageFont.load_default()

    # color[::-1] để đảo ngược RGB thành BGR
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ================================================================================================================================
# LOGIC NHẬN DIỆN


def process_fire_logic(
    frame, motion_mask, mode="webcam", frame_count=0, fps=30, alert_times=None
):
    # Tiền xử lý
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Thiết lập ngưỡng tùy theo chế độ Webcam/Upload
    if mode == "webcam":
        lower_fire = np.array([0, 120, 180])
        upper_fire = np.array([35, 255, 255])
        dilate_iter = 2
        extent_range = (0.15, 0.65)
        ratio_range = (0.3, 1.3)
        solidity_max = 0.88
        texture_threshold = 22
    else:
        lower_fire = np.array([0, 70, 200])
        upper_fire = np.array([25, 255, 255])
        dilate_iter = 2
        extent_range = (0.2, 0.75)
        ratio_range = (0.4, 2.5)
        solidity_max = 1.1
        texture_threshold = 18

    # Phân đoạn ảnh
    mask_color = cv2.inRange(hsv, lower_fire, upper_fire)
    thres = cv2.bitwise_and(mask_color, mask_color, mask=motion_mask)
    kernel = np.ones((5, 5), dtype="uint8")
    dilated = cv2.dilate(thres, kernel, iterations=dilate_iter)
    dilated_eroded = cv2.erode(dilated, kernel, iterations=1)

    # Tìm & Lọc đối tượng
    contours, _ = cv2.findContours(
        dilated_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)

        # Chỉ xét vùng có diện tích đủ lớn (>150 pixel)
        if area > 150:
            x, y, w, h = cv2.boundingRect(contour)

            # Kiểm tra hình dáng lửa
            rect_area = w * h
            extent = float(area) / rect_area
            aspect_ratio = w / float(h)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Kiểm tra kết cấu
            roi_gray = gray[y : y + h, x : x + w]
            if roi_gray.size > 0:
                texture_score = np.std(roi_gray)
            else:
                texture_score = 0

            # Logic tổng hợp để quyết định nếu thỏa mãn tất cả điều kiện
            if (
                (extent_range[0] < extent < extent_range[1])
                and (ratio_range[0] < aspect_ratio < ratio_range[1])
                and (solidity < solidity_max)
                and (texture_score > texture_threshold)
            ):
                # Vẽ cảnh báo
                color = (0, 0, 255)
                label = "Cảnh báo Cháy!"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                frame = draw_vietnamese_text(frame, label, (x, y - 25), color)

                # Ghi lại thời điểm cháy để chèn âm thanh
                if alert_times is not None:
                    alert_time = frame_count / fps
                    if not alert_times or alert_time - alert_times[-1] > 0.4:
                        alert_times.append(alert_time)

    return frame


# ================================================================================================================================
# XỬ LÝ VIDEO


def process_video_final(
    video_webcam, video_upload, current_mode, progress=gr.Progress()
):
    # Khởi tạo biến
    cap = None
    out = None

    # Xác định nguồn video
    if current_mode == "webcam":
        video_path = video_webcam
        file_name_goc = "Webcam"
        print(f"Đang xử lý chế độ: WEBCAM")
    else:
        video_path = video_upload
        current_mode = "video_file"
        print(f"Đang xử lý chế độ: UPLOAD")

        # Lấy tên file gốc để đặt tên file kết quả
        if video_upload is not None:
            file_name_goc = os.path.splitext(os.path.basename(video_upload))[0]
        else:
            file_name_goc = "Unknown"
    if video_path is None:
        return None

    # Tạo thư mục lưu kết quả và đặt tên các file video
    OUTPUT_DIR = "VideoKetQua"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    temp_filename = f"Temp_{file_name_goc}.mp4"
    temp_path = os.path.join(OUTPUT_DIR, temp_filename)
    final_filename = f"VideoKetQua_{file_name_goc}.mp4"
    final_path = os.path.join(OUTPUT_DIR, final_filename)

    # Thuật toán trừ nền để tách vật thể chuyển động khỏi background tĩnh
    back_sub_video = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    alert_times = []
    frame_count = 0

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Không thể mở video")
            return None

        # Lấy FPS và tổng số frame để hiện thanh loading
        fps_input = cap.get(cv2.CAP_PROP_FPS)
        if fps_input is None or fps_input <= 0 or fps_input > 120:
            fps_input = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Giảm kích thước ảnh để xử lý nhanh hơn
            h_orig, w_orig = frame.shape[:2]
            if w_orig > 640:
                scale = 640 / w_orig
                frame = cv2.resize(frame, (640, int(h_orig * scale)))

            # Lấy kích thước mới sau khi resize để tạo VideoWriter
            h_new, w_new = frame.shape[:2]

            # Khởi tạo VideoWriter (chỉ làm 1 lần ở frame đầu tiên)
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_path, fourcc, fps_input, (w_new, h_new))
                if not out.isOpened():
                    temp_path = temp_path.replace(".mp4", ".avi")
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    out = cv2.VideoWriter(temp_path, fourcc, fps_input, (w_new, h_new))

            frame_count += 1

            # Cập nhật tiến trình mỗi 10 frame để đỡ lag UI
            if frame_count % 10 == 0:
                progress((frame_count, total_frames), desc="Đang xử lý hình ảnh...")

            # Áp dụng trừ nền để lấy mặt nạ chuyển động
            motion_mask = back_sub_video.apply(frame)

            # Bỏ qua 30 frame đầu để ổn định background
            if frame_count < 30:
                if out is not None:
                    out.write(frame)
                continue

            # Khử nhiễu cho mặt nạ chuyển động
            kernel_motion = np.ones((3, 3), dtype="uint8")
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_motion)

            # Gọi hàm xử lý logic ở trên
            processed_frame = process_fire_logic(
                frame,
                motion_mask,
                mode=current_mode,
                frame_count=frame_count,
                fps=fps_input,
                alert_times=alert_times,
            )

            if out is not None:
                out.write(processed_frame)

    except Exception as e:
        print(f"Lỗi OpenCV: {e}")
        import traceback

        traceback.print_exc()

    # Khi hoàn tất thì giải phóng tài nguyên để kh tràn RAM
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

    if not os.path.exists(temp_path):
        return None

    # Xử lý âm thanh
    try:
        progress(0.9, desc="Đang ghép âm thanh...")
        print("Đang xử lý ghép âm thanh...")

        # Nếu có cháy thì tiến hành ghép âm thanh cảnh báo và xuất file kết quả
        if alert_times:
            # Dùng MoviePy để chèn file beep.wav vào đúng giây phát hiện cháy
            video_clip = mp.VideoFileClip(temp_path)
            beep_clip = mp.AudioFileClip("beep.wav")
            beeps = [beep_clip.set_start(t) for t in alert_times]

            # Trộn âm thanh gốc của video (nếu có) với tiếng bíp
            if video_clip.audio is not None:
                final_audio = mp.CompositeAudioClip([video_clip.audio] + beeps)
            else:
                final_audio = mp.CompositeAudioClip(beeps)

            final_video = video_clip.set_audio(final_audio)

            # Tối ưu để xuất file cuối cùng
            final_video.write_videofile(
                final_path,
                codec="libx264",
                audio_codec="aac",
                logger=None,
                preset="ultrafast",
                threads=4,
            )
            video_clip.close()
            beep_clip.close()

            # Xóa file tạm không có tiếng
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Nếu không có cháy thì chỉ cần đổi tên file tạm thành file kết quả
        else:
            if os.path.exists(final_path):
                os.remove(final_path)
            if temp_path.endswith(".avi") and final_path.endswith(".mp4"):
                final_path = final_path.replace(".mp4", ".avi")
            os.rename(temp_path, final_path)

        return final_path

    except Exception as e:
        print(f"Lỗi ghép âm thanh: {e}")
        return temp_path


# ================================================================================================================================
# GIAO DIỆN GRADIO


# Hàm chuyển sang giao diện Webcam
def switch_to_webcam():
    return gr.update(visible=True), gr.update(visible=False, value=None), "webcam"


# Hàm chuyển sang giao diện Upload
def switch_to_upload():
    return gr.update(visible=False, value=None), gr.update(visible=True), "upload"


# Hàm xóa tất cả và reset về trạng thái ban đầu
def clear_all_data(current_mode):
    if current_mode == "webcam":
        return (
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=False),
            None,
            "webcam",
        )
    else:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=True),
            None,
            "upload",
        )


# Tạo bố cục trang web
with gr.Blocks(title="HỆ THỐNG CẢNH BÁO CHÁY") as app:
    # Tiêu đề
    gr.HTML(
        """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #F97E39; font-size: 2em;">🔥 HỆ THỐNG CẢNH BÁO CHÁY TỰ ĐỘNG 🔥</h1>
    </div>
    """
    )

    # Biến ngầm để nhớ đang chọn chế độ nào
    mode_state = gr.State(value="upload")

    # Bố cục hàng/cột trong trang
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                btn_mode_up = gr.Button("Tải File Video", variant="primary")
                btn_mode_cam = gr.Button("Quay Webcam", variant="primary")

            inp_upload = gr.Video(
                sources=["upload"],
                label="Hãy bấm vào nền để bắt đầu",
                visible=True,
                height=535,
            )

            inp_webcam = gr.Video(
                label="Hãy bấm vào biểu tượng Camera để bắt đầu",
                visible=False,
                height=535,
            )

        with gr.Column(scale=1):
            with gr.Row():
                btn_reset = gr.Button("Làm Mới", variant="secondary")
                btn_process = gr.Button("BẮT ĐẦU XỬ LÝ", variant="stop")

            output_video = gr.Video(label="Video Kết Quả", height=535)

    btn_mode_cam.click(
        fn=switch_to_webcam, inputs=None, outputs=[inp_webcam, inp_upload, mode_state]
    )

    btn_mode_up.click(
        fn=switch_to_upload, inputs=None, outputs=[inp_webcam, inp_upload, mode_state]
    )

    btn_process.click(
        fn=process_video_final,
        inputs=[inp_webcam, inp_upload, mode_state],
        outputs=output_video,
    )

    btn_reset.click(
        fn=clear_all_data,
        inputs=[mode_state],
        outputs=[inp_webcam, inp_upload, output_video, mode_state],
    )

if __name__ == "__main__":
    app.launch()
