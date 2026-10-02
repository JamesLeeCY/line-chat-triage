from datetime import datetime

from src.parser import load_employees, merge_fragments, parse_file

SAMPLE = (
    "[LINE] 測試群組\n"
    "儲存日期：2025/09/05 17:00\n"
    "\n"
    "2025.09.01 星期一\n"
    "09:00\t客戶A\t請問進度如何？\n"
    "09:00\t客戶A\t有點急\n"
    "09:05\t成員1\t今天會回覆您\n"
    "09:06\t客戶A\t貼圖\n"
    "09:07\t系統訊息\t系統訊息不該被解析\n"
    "這行不是訊息格式\n"
    "2025.09.02 星期二\n"
    "10:30\t成員1\t圖片\n"
)


def _write(tmp_path, text, name="chat.txt"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_employees_skips_blank_and_comments(tmp_path):
    path = _write(tmp_path, "# 註解\n成員1\n\n  成員2  \n", "emp.txt")
    assert load_employees(path) == {"成員1", "成員2"}


def test_parse_file_roles_dates_and_media(tmp_path):
    group_id, msgs = parse_file(_write(tmp_path, SAMPLE), {"成員1"})

    assert group_id == "測試群組"
    assert [m.sender for m in msgs] == ["客戶A", "客戶A", "成員1", "客戶A", "成員1"]
    assert [m.role for m in msgs] == ["customer", "customer", "staff", "customer", "staff"]
    assert msgs[0].timestamp == datetime(2025, 9, 1, 9, 0)
    assert msgs[-1].timestamp == datetime(2025, 9, 2, 10, 30)

    sticker, image = msgs[3], msgs[4]
    assert sticker.is_sticker and sticker.text == ""
    assert image.is_image and image.text == ""
    assert len({m.msg_id for m in msgs}) == len(msgs)


def test_parse_file_falls_back_to_filename(tmp_path):
    group_id, _ = parse_file(_write(tmp_path, "2025.09.01 星期一\n09:00\tA\thi\n", "某群組.txt"), set())
    assert group_id == "某群組"


def test_merge_fragments_joins_same_sender_within_window(tmp_path):
    _, msgs = parse_file(_write(tmp_path, SAMPLE), {"成員1"})
    merged = merge_fragments(msgs, window_secs=60)

    assert merged[0].text == "請問進度如何？ 有點急"
    assert merged[0].msg_id == msgs[0].msg_id
    # Stickers are never merged into text
    assert any(m.is_sticker for m in merged)
    assert len(merged) == len(msgs) - 1


def test_merge_fragments_respects_window(make_msg):
    msgs = [
        make_msg("客戶A", "customer", "2025-09-01T09:00", "第一句"),
        make_msg("客戶A", "customer", "2025-09-01T09:05", "五分鐘後"),
    ]
    assert len(merge_fragments(msgs, window_secs=60)) == 2
