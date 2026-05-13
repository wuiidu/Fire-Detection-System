@echo off
:: Chuyển mã sang UTF-8 để hiển thị tiếng Việt
chcp 65001 >nul
title HỆ THỐNG CẢNH BÁO CHÁY TỰ ĐỘNG

:: Kiểm tra xem đã có môi trường ảo chưa, nếu chưa thì tạo mới
if not exist MoiTruongAo (
    echo Phát hiện lần chạy đầu tiên. Đang tạo môi trường ảo...
    python -m venv MoiTruongAo
)

:: Kiểm tra và cài đặt/cập nhật thư viện từ ThuVien.txt
echo Đang kiểm tra và cập nhật các thư viện cần thiết...
echo ----------------------------------------------------------------------------------------------------

.\MoiTruongAo\Scripts\pip install -r ThuVien.txt

echo ----------------------------------------------------------------------------------------------------
echo Cập nhật thư viện hoàn tất.
:: Chạy chương trình
echo Đang khởi động Hệ Thống Cảnh Báo Cháy Tự Động...
echo ----------------------------------------------------------------------------------------------------

:: Chạy file PhatHienChay.py
.\MoiTruongAo\Scripts\python PhatHienChay.py

:: Giữ màn hình không bị tắt để xem nếu có lỗi
pause