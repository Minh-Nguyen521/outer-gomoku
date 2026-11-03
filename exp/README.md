# Training experiments: 4 quick methods + combined pipeline

Mỗi thư mục là một phương pháp độc lập, đã smoke-tested, chạy được ngay trên
M2 (MPS) hoặc Kaggle T4. Mục tiêu: tìm hướng đi nhanh trước khi đầu tư
self-play full-scale.

| Folder | Phương pháp | Chi phí | Trả lời câu hỏi |
|---|---|---|---|
| `01_supervised/` | SL thuần trên harvest data | phút–giờ | Imitation lấy ta tới đâu? (baseline) |
| `02_exit_iter/` | Expert Iteration (SL + self-play, không search) | giờ | Bootstrap từ ván tự chơi có hơn SL tĩnh? |
| `03_distill/` | KataGomo teacher labels (policy distribution) | giờ (cần KataGomo) | Teacher distribution >> one-hot bao nhiêu? |
| `04_gumbel/` | Gumbel AZ trên 9×9 (hoặc 15×15) | giờ–ngày | Pipeline AlphaZero-style có học không? |
| `pipeline/` | **Chuỗi 3→2→4 với warm-start** | qua đêm | Kết hợp tất cả thành một model duy nhất |

## Pipeline (khuyến nghị chính)

    exp/pipeline/run.sh        # distil -> ExIt -> Gumbel AZ, tự skip stage đã xong
    exp/pipeline/run.sh 13      # bỏ ExIt

Mỗi stage warm-start bằng checkpoint của stage trước (cùng kiến trúc
`tools/train_policy.Net`, 6-plane 15×15) nên không phí công học lại.

## So sánh kết quả

Sau khi train từng cái, đấu chéo bằng `tools/match.py`:

    # net vs net
    python3 tools/match.py --games 20 --parallel 1 --rule 0 --timeout-turn 10000 \
        --engine-a "python3 oog_az/player.py --model exp/01_supervised/supervised.pt" \
        --engine-b "python3 oog_az/player.py --model exp/03_distill/distilled.pt"

    # net vs ladder engine (yardstick ngoài)
    python3 tools/match.py ... --engine-b third_party/SlowRenju/pbrain-slowrenju

## Gợi ý thứ tự chạy

1. `01_supervised/run.sh` — rẻ nhất, cho baseline metrics (val top1, vMAE)
2. `03_distill/run.sh` — dùng lại data harvest, chỉ thêm bước KataGomo label
3. `02_exit_iter/run.sh` — 5 iterations là thấy trend
4. `04_gumbel/run.sh` với `--size 9` — prototype AZ; scale 15 sau khi ỏn

## Lưu ý

- `oog_az/player.py` cần mô hình có format checkpoint tương thích
(`load_checkpoint`); các net train bằng `tools/train_policy.Net` (6-plane)
khác 5-plane AZNet — nếu muốn đánh giá qua player.py, cần chuyển đổi plane
hoặc dùng bản 02 (nó cùng kiến trúc Net 6-plane).
- Data càng nhiều (Kaggle harvest đang chạy), 01/02/03 càng mạnh.
- Kết luận từ `docs/value-head.md`: đừng hy vọng value head học được từ
harvest data (contaminated) — hãy nhìn metric policy là chính; value chỉ
ổn trong 02 (self-play) và 04.
