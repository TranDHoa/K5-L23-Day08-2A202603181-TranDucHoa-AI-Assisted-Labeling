# Thông tin học viên

- Họ và tên: Trần Đức Hòa
- MSHV: 2A202603181
- Lớp (L2 / L3 / khác): L2A
- Tài khoản GitHub: TranDHoa
- Repo bài nộp (URL): https://github.com/TranDHoa/K5-L23-Day08-2A202603181-TranDucHoa-AI-Assisted-Labeling.git
- Commit chốt bài (hash): 9867e5f
- Máy: Windows (WSL2 + Docker Desktop v29.9.2 / Compose v5.6.1), Python 3.12 (.venv), chạy inference trên CPU
- CVAT (Docker): CVAT v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ submission/qc/*, kèm mẫu số)

- Máy: accepted / edited / deleted: 417/622 (67.0%) / 59/622 (9.5%) / 145/622 (23.3%) (mẫu số: 622 box máy); added: 85/561 (15.2%) (mẫu số: 561 đối tượng sau duyệt)
- So với gold: precision / recall của **máy**: 57/83 = 68.7% (KTC 95%: 58–78%) / 57/65 = 87.7% (KTC 95%: 78–94%)
- Bắt bẫy: 11/12 = 91.7% (KTC 95%: 65–99%)
- Thời gian: gán tay 5.5 phút/ảnh (22 phút / 4 ảnh); duyệt có pre-label 2.8 phút/ảnh (67 phút / 24 ảnh)
- Lỗi QC: 0.0% theo ảnh (0/7), 0.0% theo đối tượng (0/137)
- **Quyết định:** Release — lý do một dòng: Đã hoàn tất Rework từ vòng 1, đạt 11/12 (91.7% >= 80%) bẫy, 0/7 (0.0% <= 2%) lỗi QC trên mẫu, 0 cảnh báo alidate_labels.py và chuẩn hóa guideline v1.1 (CAL-01 đến CAL-04).