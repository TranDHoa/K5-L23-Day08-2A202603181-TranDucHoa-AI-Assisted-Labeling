# Báo cáo Day 08 — 2A202603181 - Trần Đức Hòa

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | `yolo26-m` (chạy toàn bộ 36 ảnh ở `conf=0.05` lưu tại `work/prelabel/y26_all/pre.coco.json` để quét ngưỡng; pilot `r01` `conf=0.25` và `r02` `conf=0.5`) | `bicycle=0.2`, `motorbike=0.1`, các lớp còn lại (`person`, `car`, `bus`, `truck`) dùng ngưỡng mặc định `0.3` | Từ `r01` (`215` boxes/8 ảnh) và `r02` (`139` boxes/8 ảnh) cho thấy `yolo26-m` đạt tốc độ nhanh trên CPU (`0.51s/ảnh` trên 36 ảnh) và ôm biên rất chuẩn cho xe phổ thông. Khi quét ngưỡng trên 4 ảnh gold (mục tiêu recall `>= 90%`): `bicycle` đạt recall `100%` (5/5 vật) và precision `83%` tại `0.2`; `motorbike` đạt recall `94%` (17/18 vật) và precision `71%` tại `0.1`; `person` (tối đa `88%`, 26 vật) và `car` (tối đa `88%`, 8 vật) không đạt mốc `90%` ở `<= 0.5`, còn `bus` (1 vật) và `truck` (0 vật) có `< 5` mẫu gold nên giữ ngưỡng mặc định `0.3`. |
| three_wheeler | `yoloe-s` (`r03`, chạy 36 ảnh tại `work/prelabel/yoloe_all/pre.coco.json` với 3 text prompts: `'tuk-tuk'`, `'cyclo'`, `'cargo tricycle'`) | `three_wheeler=0.15` | Ở `r01`/`r02`, `yolo26-m` bỏ sót hoàn toàn xe tuk-tuk và xích lô. `yoloe-s` (`r03`, `0.30s/ảnh` trên CPU) giải quyết triệt để điểm mù này; qua quét ngưỡng trên 4 ảnh gold (7 vật `three_wheeler`), ngưỡng `0.15` đạt recall `100%` (7/7 vật) và precision `78%`. |

- **Lệnh `merge_prelabels.py` đã dùng (dán nguyên):** python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "bicycle=0.2,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn SAM 2.1 (sam2.1_b.pt) cho tác vụ phân vùng đối tượng chính (Part 5) vì chất lượng đường biên mask sắc nét hơn hẳn MobileSAM (mobile_sam.pt), tách bạch tốt người lái khỏi thân xe khi kết hợp điểm negative (--neg) và chạy ổn định trên cấu hình CPU cục bộ (không cần GPU VRAM lớn hay xin quyền truy cập API bên ngoài); tuy nhiên ở bước pre-label 2D BBox (r04), yolo26-m + sam2.1_b.pt chạy chậm (5.12s/ảnh trên CPU) và dễ bị tràn mask qua khung xe hở nên đã loại khỏi pipeline tạo box.
- Lỗi nào **không sửa được bằng đổi ngưỡng**: 1. Mô hình nhận diện thiếu lớp hoặc nhầm lớp: yolo26-m không có lớp three_wheeler trong tập huấn luyện nên có thể bỏ sót xe ba bánh (xích lô, tuk-tuk) hoặc nhận nhầm sang các lớp khác như car, truck. Việc điều chỉnh ngưỡng confidence không thể tạo ra lớp mới. Cách sửa: Sử dụng mô hình open-vocabulary yoloe-s với 3 prompt (tuk-tuk, cyclo, cargo tricycle) để nhận diện riêng lớp three_wheeler, sau đó gộp kết quả với yolo26-m thông qua merge_prelabels.py.

2. Bounding box giãn biên (loose): Với các đối tượng có cấu trúc khung hở, bounding box dễ bị kéo quá rộng, bao gồm cả vùng nền hoặc lấn sang đối tượng khác, làm giảm độ chính xác của annotation. Cách sửa: Kiểm tra và chỉnh sửa thủ công trên CVAT, kéo khít 4 cạnh bounding box theo quy tắc CAL-01.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): Ở img_015.jpg, yolo26-m sinh ra box ảo bicycle 0.80 khoanh nhầm vào khung cửa sổ trong bóng râm; Ở img_032.jpg, chiếc xích lô ở hậu cảnh bên trái được yoloe-s bắt chính xác với three_wheeler 0.27; ở img_002.jpg
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Chọn ngưỡng trên chính 4 ảnh gold (in-sample, tổng 65 vật thể) nên các chỉ số Precision (68.7%) và Recall (87.7%) của máy trên gold là con số lạc quan (có hiện tượng optimistic bias), đồng thời cỡ mẫu nhỏ khiến khoảng tin cậy 95% Wilson rất rộng (58–78% cho Precision và 78–94% cho Recall) và không đủ mẫu để quét ngưỡng riêng cho bus (1 vật) và truck (0 vật).

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 417 | 67 | 622 |
| edited | 59 | 9.5 | 622 |
| deleted | 145 | 23.3 | 622 |
| added | 85 | 15.2 | 561 |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? 
  - Xét theo tỉ lệ bị xóa (deleted) và chỉnh sửa (edited), three_wheeler lớp máy gây tốn công dọn dẹp nhất:

  - three_wheeler: Có tới 27 box bị xóa và 5 box phải sửa trong khi chỉ có 10 box được giữ nguyên (accepted chỉ đạt 10/42 = 23.8% tổng box máy của lớp này). Nguyên nhân do đặt ngưỡng thấp (0.15) khiến yoloe-s sinh nhiều box nhiễu vào quầy hàng vỉa hè (img_002.jpg), đuôi xe tải (img_011.jpg) hoặc kéo box trùm sang xe kế bên (img_013.jpg, img_017.jpg).
- So với gold: Precision của máy là 57/83 = 68.7% (KTC 95% Wilson: 58–78%, mẫu số 83 dự đoán của máy trên gold: 57 TP + 26 FP); Recall của máy là 57/65 = 87.7% (KTC 95% Wilson: 78–94%, mẫu số 65 đối tượng gold: 57 TP + 8 FN).
## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: Ở vòng duyệt đầu tiên chỉ bắt được 8/12 = 66.7% (KTC 95% Wilson: 39–86%, chưa đạt ngưỡng 80%), lọt 4 bẫy gồm img_025.jpg (missing), img_034.jpg (missing), img_021.jpg (loose) và img_003.jpg (loose). Sau khi siết lại quy trình quét ảnh và duyệt lại lô (Rework), kết quả cuối cùng đạt 11/12 = 91.7% (KTC 95% Wilson: 65–99%, vượt ngưỡng 80%), bắt trọn wrong_class (3/3), extra (3/3), loose (3/3) và chỉ còn lọt duy nhất 1 bẫy missing ở img_034.jpg. Nguyên nhân để lọt ở vòng 1 là do bỏ qua bước tự quét mắt toàn ảnh trước khi bật hiển thị box máy và quá tin vào độ khít biên của các box có sẵn (automation bias).
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? Ở vòng duyệt đầu, tỉ lệ accepted khá cao (432/622 = 69.5%) nhưng tỉ lệ bắt bẫy chỉ đạt 8/12 = 66.7%: điều này phản ánh rõ thiên kiến tự động hóa (automation bias) — người duyệt có xu hướng bấm chấp nhận nhanh các box máy vẽ sẵn mà không kiểm tra kỹ 4 cạnh biên (để lọt bẫy loose ở img_003.jpg, img_021.jpg) và chỉ tập trung xóa/sửa nơi có box hiển thị sẵn mà quên quét các vùng trống ở rìa ảnh hoặc tuyến sau (để lọt bẫy missing ở img_025.jpg, img_034.jpg). Khi áp dụng quy trình quét ảnh độc lập trước khi xem pre-label ở vòng 2, tỉ lệ bắt bẫy tăng lên 11/12 (91.7%) và accepted giảm nhẹ xuống 67.0% (417/622) do đã chủ động chỉnh sửa nhiều box lỏng hơn.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): Gán tay hoàn toàn từ đầu trên 4 ảnh gold mất khoảng 6.5 phút/ảnh (26 phút / 4 ảnh), trong khi duyệt và sửa trên nền AI pre-label (batch_review, 24 ảnh) mất khoảng 2.8 phút/ảnh (67 phút / 24 ảnh, nhanh hơn khoảng 2.3 lần). Thiên lệch: Có tồn tại thiên lệch quen cảnh (familiarity bias) vì người duyệt đã tự tay gán 4 ảnh gold và xem qua 8 ảnh pilot trước đó nên đã nắm rõ taxonomy v1.0 và các tình huống giao thông đặc trưng, giúp tốc độ duyệt lô sau nhanh hơn so với người mới tiếp cận lần đầu.
- Kết luận: Có đáng dùng, đặc biệt hiệu quả cao cho các lớp phương tiện và người phổ thông có độ chính xác cao từ yolo26-m (person, car, bus, motorbike tiền cảnh — giúp giữ nguyên được 417/622 = 67.0% số box mà không phải vẽ lại từ đầu). Tuy nhiên với lớp three_wheeler (yoloe-s ngưỡng 0.15), máy chỉ đóng vai trò gợi ý vị trí ban đầu vì tỉ lệ xóa/sửa lên tới 32/42 = 76.2% box máy của lớp này.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. Quét ảnh độc lập trước khi bật nhãn máy : Tắt/ẩn lớp hiển thị pre-label trên CVAT (phím tắt ẩn nhãn) và quét mắt toàn khung hình theo lưới từ trái sang phải, từ tiền cảnh ra hậu cảnh để đếm nhẩm vật thể trước khi bật box máy kiểm tra (giảm lỗi missing từ 2/3 lọt xuống 1/3 lọt).
  2. Kiểm tra 4 cạnh biên cho mọi box giữ lại : Không bấm accepted theo cảm tính; mọi box three_wheeler, car, motorbike đều phải phóng to kiểm tra 4 cạnh đảm bảo không thừa nền đường quá 3–5px và không trùm sang xe kế bên (nâng bắt bẫy loose từ 1/3 lên 3/3)
  3. Kiểm tra kích thước tối thiểu 20px trước khi duyệt xong một ảnh : Soi kỹ các cụm xe/người ở cuối đường tuyến xa, chuyển toàn bộ vật thể có chiều cao < 20px thành ignore_region và chạy validate_labels.py xác nhận 0 cảnh báo trước khi chốt tệp.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (đã tách được tài xế nhờ điểm neg `720,300`) | Phần khoang mở bị thủng lỗ chỗ ở các thanh khung sắt mảnh; mask hơi tràn nhẹ qua khe hở ra khung cảnh phía sau | `tuktuk_img013_sam2.jpg` (và `.json` nếu có) |
| `img_013.jpg` | MobileSAM (`mobile_sam.pt`) | 3 (2 pos, 1 neg) | Có dính một phần áo/tay tài xế | Biên mask răng cưa hơn SAM 2.1, khó tách dứt khoát giữa phần mui/thân xe đỏ với áo tài xế và bánh xe phía sau | `tuktuk_img013_sam1.jpg` |
| `img_004.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (điểm neg `465,450` loại được người đứng cạnh bên trái) | Tách tốt khối đầu và thân xe tuk-tuk xanh dương tiền cảnh, nhưng vẫn bị mất một vài nan sắt mảnh ở khoang chở khách | `tuktuk_img004_sam2.jpg` |
| `img_017.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 (2 pos, 1 neg) | Không (điểm neg `645,355` chặn được phần người/hàng hóa bên trong) | Tách gọn phần thân vỏ xe nhưng dễ bị đứt đoạn giữa phần mui bạt phía trên và thùng xe phía dưới nếu chỉ dùng ít điểm pos | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2:
  - **Quy tắc giúp ích:** Quy định rõ ràng về việc **tách biệt người lái / hành khách (`person`) và hàng hóa rời khỏi mask của phương tiện (`three_wheeler`)** giúp đặt điểm negative (`--neg`) đúng trọng tâm (vào lưng tài xế hoặc người đứng sát xe), tránh tình trạng SAM gộp chung cả người ngồi bên trong vào khối phương tiện. Quy tắc ưu tiên phần thân vỏ đặc của xe cũng giúp chọn vị trí đặt điểm positive (`--pos`) ổn định thay vì click vào các thanh sắt mảnh.
  - **Quy tắc còn thiếu:** Guideline chưa quy định chi tiết cách xử lý các **khoang rỗng / khung sắt hở (see-through structures)** đặc trưng của xe tuk-tuk và xích lô — cụ thể là phần không gian nhìn xuyên thấu qua khung xe (thấy nền đường, cây cối hoặc người đi bộ phía sau) nên để **thủng (multi-polygon / hole)** theo đúng thực tế pixel hay vẽ **bao trùm (amodal/convex)** qua cả khung hở, cũng như chưa chốt ngưỡng diện tích tối thiểu của các mảnh mask rời (ví dụ phần mui bạt bị ngăn cách bởi thanh chống mảnh) có được phép giữ lại hay không.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Chưa kích hoạt/sử dụng trực tiếp trong bản CVAT Docker local (do không cài thêm container Nuclio serverless functions cho SAM); toàn bộ mask được tạo qua script offline `sam2_points.py` trên CPU. So với dùng trực tiếp trên giao diện CVAT, chạy offline bằng CLI cho phép tái lập chính xác tọa độ các điểm `--pos` / `--neg` và so sánh đối chứng nhanh giữa hai trọng số (`sam2.1_b.pt` vs `mobile_sam.pt`) trên cùng bộ tọa độ click, nhưng mất thời gian dò tọa độ pixel thủ công hơn so với click trực quan trên web.
## 6. Calibration (Part 6)

- **Số quyết định:** **4** quyết định (`CAL-01` đến `CAL-04` trong `calibration_decisions.csv`); quan trọng nhất là **`CAL-01`** (chuẩn hóa phân loại và quy tắc đóng khung lớp `three_wheeler` kết hợp bổ sung prompt từ `yoloe-s` với `conf >= 0.15`, giải quyết triệt để điểm mù của `yolo26-m` đối với xe tuk-tuk, xích lô và xe ba gác đặc thù giao thông châu Á) cùng **`CAL-04`** (quy trình quét mắt toàn ảnh trước khi bật hiển thị nhãn máy để chống thiên kiến tự động hóa *automation bias*, giúp nâng tỉ lệ bắt bẫy từ `8/12 = 66.7%` lên `11/12 = 91.7%`).
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:**
  - **Mục 2.1 & Mục 3.1 (`CAL-01`):** Quy định gán toàn bộ xe tuk-tuk, xích lô, xe ba gác chở hàng vào lớp `three_wheeler`; đóng khung ôm sát thân/mái che/bánh xe từng chiếc độc lập (không kéo trùm sang xe nối đuôi bên cạnh) và tách riêng người lái/hành khách thành `person` (`img_002.jpg`, `img_003.jpg`, `img_013.jpg`, `img_017.jpg`, `img_032.jpg`).
  - **Mục 2.8 (`CAL-02`):** Thiết lập ngưỡng kích thước tối thiểu `20px` — không vẽ box đối tượng đơn lẻ cho vật thể tuyến xa có chiều cao `< 20px` mà gộp thành vùng `ignore_region`, loại bỏ hoàn toàn 14 cảnh báo từ `validate_labels.py` (`img_018.jpg`, `img_033.jpg`, `img_034.jpg`).
  - **Mục 2.2 & Mục 4.1 (`CAL-03`):** Tách rời quang gánh, thúng hàng và khung xe khỏi bounding box của `person`; bật cờ `can_review = true` với các trường hợp hành khách ngồi sâu trong khoang xe bị khung sắt che khuất `> 50%` (`img_013.jpg`, `img_014.jpg`, `img_032.jpg`).
  - **Mục 5 (`CAL-04`):** Bắt buộc quét ảnh từ trái sang phải trước khi xem pre-label của máy, xóa sạch box đè trùng lặp do hạ ngưỡng `motorbike=0.1` và siết chặt 4 cạnh box vào đúng đường biên vật thể (`img_003.jpg`, `img_006.jpg`, `img_021.jpg`, `img_025.jpg`, `img_034.jpg`).
- **Nguyên nhân gốc:** Kết hợp từ cả ba yếu tố: **(1) Máy (Domain shift & cấu hình ngưỡng):** `yolo26-m` huấn luyện trên tập COCO không nhận diện được xe ba bánh (`three_wheeler`), trong khi hạ sâu ngưỡng `motorbike=0.1` và `three_wheeler=0.15` để đạt mục tiêu recall `90%` làm phát sinh nhiều box đè và box giãn biên; **(2) Guideline v1.0 mơ hồ ở một số ca biên:** chưa chốt rõ quy tắc tách hàng hóa cồng kềnh khỏi `person` và cách gom cụm vật thể `< 20px` ở tuyến xa thành `ignore_region`; **(3) Thao tác người duyệt chưa chặt ở vòng 1:** thiên kiến tự động hóa (*automation bias*) khiến dễ bỏ qua bẫy box lỏng (`loose`) và vật thể bị che khuất ở rìa ảnh (`missing`) trước khi siết lại checklist ở vòng 2.

## 7. QC và quyết định (Part 6)

- Mẫu: 7 ảnh (2 ảnh can_review + 5 ngẫu nhiên, seed 42), tổng cộng 137 đối tượng theo taxonomy v1.0; người kiểm: Trần Đức Hòa
- Lỗi theo **đối tượng**: 0/137 = 0.0% · theo **ảnh**: 0/7 = 0.0% (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: 
  -Trên tệp qc_review.csv cuối cùng sau khi chỉnh sửa: không ghi nhận lỗi tồn đọng (0/137 đối tượng).

  -Trong quá trình rà soát trước khi chốt bản cuối, các nhóm lỗi phổ biến nhất được phát hiện và khắc phục gồm:

  - Thừa : Xảy ra nhiều ở motorbike và bicycle do ngưỡng thấp, ví dụ img_006.jpg (box #71 đè lên xe đạp #97), img_014.jpg (box #144, #145 đè cụm xe máy bên phải), img_035.jpg (box #575, #576 đè xe máy trên vỉa hè).

  - Sai vị trí / kích thước: Xảy ra ở các xe ba bánh three_wheeler có khoang rỗng hoặc đỗ nối đuôi, ví dụ img_003.jpg (box #21 thừa khoảng trống trên mui và dưới mặt đường), img_013.jpg (box #138 trùm sang xe phía xa), img_017.jpg (box #160 kéo trùm sang xe tuk-tuk phía trước).

  - Sai lớp: Ví dụ img_018.jpg (8 box cao 14.3–18.8px), img_033.jpg (box #505, #497 cao < 20px), img_034.jpg (box #511 cao 17.4px).

  -Bỏ sót: Vật thể bị che khuất ở rìa/tuyến sau, ví dụ img_025.jpg và img_034.jpg (1 bẫy missing còn lọt trong thống kê tổng 11/12 bẫy).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Với 0 ảnh lỗi trên 7 ảnh mẫu, cận trên khoảng tin cậy 95% Wilson của tỉ lệ ảnh lỗi thật vẫn lên tới 35% (và với 11/12 bẫy thì KTC 95% trải rộng từ 65% đến 99%). Do đó, ngưỡng chất lượng ≤ 2% không thể được chứng minh chỉ bằng cỡ mẫu 7 ảnh mà phải kết hợp với việc vượt qua 100% kiểm tra tự động (validate_labels.py: 0 lỗi, 0 cảnh báo) và tuân thủ checklist guideline_v1.1_calibrated.md.
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): 
  - Kiểm được: Cấu trúc schema COCO, tính hợp lệ của category_id, giới hạn tọa độ box nằm trong khung ảnh, ngưỡng chiều cao tối thiểu 20px (validate_labels.py), so khớp IoU với gold_manual.json và đối chiếu tọa độ/nhãn với bộ 12 bẫy nhân tạo (review_stats.py).
  - Không kiểm được: Độ ôm sát thực tế của 4 cạnh box vào đường biên pixel vật thể (trừ 3 ca bẫy loose), các vật thể thật bị cả máy và người cùng bỏ sót, chất lượng đường biên mask của SAM và việc người duyệt có thực sự tuân thủ quy trình tắt nhãn máy để quét ảnh trước hay không.

**Quyết định:* Release, ngày 2026-11-18. Lý do (đi theo Hold → Rework → Release):

- Không Hold: Guideline đã được chuẩn hóa rõ ràng lên bản v1.1_calibrated.md (giải quyết dứt điểm 4 ca khó CAL-01 đến CAL-04), dữ liệu ảnh gốc nguyên vẹn và toàn bộ tệp nhãn vượt qua validate_labels.py (4 ảnh gold/89 vật thể: OK; 24 ảnh batch/576 vật thể: OK, 0 cảnh báo).

- Đã hoàn tất Rework từ vòng 1: Ở vòng duyệt đầu tiên, lô dữ liệu bị giữ lại để Rework do tỉ lệ bắt bẫy chỉ đạt 8/12 = 66.7% (< 80%) và phát sinh 14 cảnh báo kích thước < 20px. Sau khi rà soát lại toàn bộ lô theo checklist mới, chuyển sạch các vật thể < 20px sang ignore_region, siết khít biên các box lỏng và nâng tỉ lệ bắt bẫy lên 11/12 = 91.7% (>= 80%), đồng thời tỉ lệ giữ nguyên box máy đạt 67.0% (417/622) và mẫu kiểm tra qc_review.csv đạt 0/7 = 0.0% lỗi, lô dữ liệu đã đủ điều kiện Release.

- Nếu Rework / Hold, việc cần làm tiếp: (Đã chuyển sang trạng thái Release sau khi hoàn tất Rework; ở lần cập nhật dataset tiếp theo sẽ rà soát bổ sung riêng 1 trường hợp bẫy missing còn sót tại img_034.jpg).

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Sử dụng trợ lý AI (Gemini) để đối chiếu ưu/nhược điểm trực quan giữa các lần chạy thử nghiệm (`t_sam`, `t_y26`, `t_y26_c05`, `t_yoloe`), gỡ lỗi cú pháp dòng lệnh PowerShell (`merge_prelabels.py`, kích hoạt `.venv`), phân tích số liệu thống kê từ `review_stats.md` và chuẩn hóa văn phong cho `guideline_v1.1_calibrated.md` cùng `REPORT.md`. Mọi con số, tọa độ box và quyết định gán nhãn trên CVAT đều được tự kiểm tra đối chiếu trực tiếp với log chạy script thực tế và hình ảnh gốc.
- **Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?** **Không.** Toàn bộ 4 ảnh tập gold (`gold_manual.json`, `89` đối tượng) được gán nhãn thủ công 100% từ đầu trên CVAT trước khi chạy bất kỳ mô hình pre-label nào (`yolo26`, `yoloe`, `sam`) và không tham khảo gợi ý từ AI trong lúc gán gold.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:**
  - **Cấu hình môi trường:** Chạy cục bộ trên máy tính cá nhân Windows (WSL2 + Docker Desktop v29.9.2 / Compose v5.6.1), môi trường Python 3.12 (`.venv`), chạy inference hoàn toàn trên **CPU** (`yolo26-m` mất `0.51s/ảnh`, `yoloe-s` mất `0.30s/ảnh`, `sam2.1_b.pt` mất `5.12s/ảnh`), công cụ gán nhãn **CVAT 2.81.0** triển khai qua Docker Compose.
  - **Vướng mắc cài đặt:** Windows Application Control chặn chạy trực tiếp `pip.exe` nên phải cài các gói (`torch`, `ultralytics`, `clip`) thông qua `python -m pip`; khi mở terminal PowerShell mới từng gặp lỗi `ModuleNotFoundError: No module named 'numpy'` do chưa kích hoạt lại môi trường ảo `..\.venv\Scripts\Activate.ps1`.
- **Dùng cache `data/cache/` thay vì tự chạy?** **Không.** Toàn bộ quá trình chạy inference trên tập pilot (8 ảnh), toàn bộ 36 ảnh (`y26_all`, `yoloe_all`), quét ngưỡng, trộn nhãn (`merge_prelabels.py`), cài bẫy (`inject_traps.py`) và chạy `sam2_points.py` đều được tự thực thi trực tiếp trên máy cá nhân.
- **Lỗi / bug script đã gặp (nếu có):**
  - Truyền thừa cờ `--inputs` khi chạy `scripts/merge_prelabels.py` gây lỗi `unrecognized arguments: --inputs` (đã khắc phục bằng cách truyền danh sách file dưới dạng tham số vị trí).
  - Thiếu kích hoạt môi trường ảo `.venv` khi mở phiên PowerShell mới dẫn đến `ModuleNotFoundError: No module named 'numpy'` khi chạy `validate_labels.py` và `review_stats.py`.
  - Cảnh báo `FutureWarning: torch.jit.load is deprecated` khi nạp trọng số `yoloe-s` (không ảnh hưởng tới kết quả xuất file `.coco.json`).
- **Một điều bạn làm khác lần sau:** Không hạ ngưỡng `motorbike` xuống quá sâu (`0.1`) chỉ để ép recall tăng thêm vài phần trăm trên 4 ảnh gold nhỏ, vì ngưỡng `0.1` sinh ra quá nhiều box đè trùng lặp (`76` box phải xóa tay) và nhiều vật thể siêu nhỏ `< 20px`; thay vào đó sẽ giữ ngưỡng cân bằng hơn (`0.25–0.30`) kết hợp lọc chiều cao tối thiểu `>= 20px` ngay sau bước `merge_prelabels.py` để tiết kiệm thời gian hậu xử lý trên CVAT.