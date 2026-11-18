# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603181 - Trần Đức Hòa

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe ba bánh tự chế, xe ba gác chở hàng, xích lô và xe tuk-tuk xếp vào lớp nào? | Gán tất cả vào lớp `three_wheeler`. Box bao trọn khung xe, mái che và bánh xe; tách riêng tài xế/hành khách thành `person` và không gộp hàng hóa rời ngoài khung xe. | `img_002.jpg`, `img_003.jpg`, `img_032.jpg` | `CAL-01` |
| 2 | Vật thể quá nhỏ ở tuyến xa (chiều cao `< 20px`) hoặc bị che khuất dày đặc xử lý thế nào? | Không vẽ box đối tượng đơn lẻ cho vật thể cao `< 20px`; khoanh một vùng `ignore_region` bao trùm cụm đối tượng quá nhỏ ở tuyến xa để tránh cảnh báo `validate_labels.py` và nhiễu khi huấn luyện. | `img_018.jpg`, `img_033.jpg`, `img_034.jpg` | `CAL-02` |
| 3 | Người bán hàng rong gánh hàng cồng kềnh hoặc ngồi sau khung sắt xe tuk-tuk thì vẽ box `person` ra sao? | Box `person` chỉ ôm sát cơ thể và trang phục của người (không bao trùm quang gánh, thúng hàng hay khung xe); bật cờ `can_review = true` nếu người bị khung xe che khuất `> 50%`. | `img_013.jpg`, `img_014.jpg`, `img_032.jpg` | `CAL-03` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_002.jpg`, `img_013.jpg`, `img_017.jpg` | Xe tuk-tuk và xích lô đỗ nối đuôi hoặc có khoang khung sắt nhìn xuyên thấu nền phía sau | `yolo26-m` bỏ sót hoàn toàn xe xích lô/tuk-tuk (hoặc nhận nhầm thành `car` conf thấp `0.34–0.42`); `yoloe-s` bắt đúng `three_wheeler` nhưng đôi khi kéo box giãn rộng trùm sang xe kế bên (`img_013.jpg`, `img_017.jpg`) | Bổ sung prompt `three_wheeler` từ `yoloe-s` (`conf >= 0.15`) khi merge; khi duyệt tay bắt buộc tách riêng từng xe nối đuôi và siết chặt 4 cạnh box vào đúng khung xe | Bổ sung vào **Mục 2.1 (Định nghĩa lớp `three_wheeler`)** và **Mục 3.1 (Độ khít bounding box)** |
| `img_018.jpg`, `img_033.jpg`, `img_034.jpg` | Cụm xe máy, xe đạp và người ở cuối đường tuyến xa có kích thước rất nhỏ (`14.3px – 19.1px`) | `yolo26-m` (khi hạ ngưỡng `motorbike=0.1`, `bicycle=0.2`) bắt cả các vật thể siêu nhỏ `< 20px` gây 14 cảnh báo từ `validate_labels.py` | Xóa nhãn đối tượng đơn lẻ đối với mọi box cao `< 20px` và gộp toàn bộ cụm vật thể tuyến xa vào vùng `ignore_region` | Bổ sung vào **Mục 2.8 (Quy tắc sử dụng `ignore_region` & ngưỡng kích thước tối thiểu `20px`)** |
| `img_013.jpg`, `img_014.jpg`, `img_032.jpg` | Hành khách ngồi trong khoang xe tuk-tuk/xích lô bị khung sắt chia cắt hoặc người gánh hàng rong | `yolo26-m` đôi khi cắt vụn box `person` chỉ lấy phần lưng áo (`img_013.jpg`) hoặc để sót khách ngồi khuất; `sam2.1_b.pt` dễ bị thủng mask ở các nan sắt mảnh | Giữ nguyên box `person` ước lượng trọn vẹn phần cơ thể người (tách rời quang gánh/thúng hàng) và gắn cờ `can_review = true` cho các ca bị che khuất `> 50%` | Bổ sung vào **Mục 2.2 (Lớp `person` & vật mang theo)** và **Mục 4.1 (Gắn cờ `can_review`)** |
| `img_003.jpg`, `img_021.jpg`, `img_025.jpg`, `img_034.jpg` | Bẫy `loose` (box giãn biên) và `missing` (sót vật thể ở mép ảnh/tuyến sau) bị lọt ở vòng duyệt đầu (`8/12` bẫy) | `yolo26-m` kết hợp `inject_traps.py` tạo box giãn rộng hoặc ẩn đối tượng; người duyệt bị thiên kiến tự động hóa (*automation bias*) nên lướt qua | Bắt buộc quét mắt toàn bộ ảnh theo lưới từ trái sang phải **trước khi** bật hiển thị nhãn máy; kiểm tra độ khít 4 cạnh của mọi box pre-label trước khi chốt | Bổ sung vào **Mục 5 (Quy trình duyệt dữ liệu & Chống thiên kiến AI pre-label)** |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt quy tắc phân loại & đóng khung `three_wheeler` tách biệt người lái; chuẩn hóa ngưỡng kích thước tối thiểu `20px` chuyển sang `ignore_region`; quy định tách quang gánh/thúng hàng khỏi `person` kèm cờ `can_review`; siết quy trình quét ảnh trước khi xem nhãn máy để chống *automation bias* | Rút ra từ lỗi bỏ sót/giãn biên xe ba bánh của `yolo26`/`yoloe` (`CAL-01`), cảnh báo `< 20px` trên `img_018`/`img_033`/`img_034` (`CAL-02`), ca che khuất/gánh hàng (`CAL-03`), và 4 ca bẫy lọt ở vòng duyệt đầu (`CAL-04`) |
