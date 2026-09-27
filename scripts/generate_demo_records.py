"""生成演示用的外部业务数据 data/external/records.csv。

真实业务中该文件由业务系统导出；此处用脚本生成一份可复现的演示数据，
覆盖配置中的全部演示用户与 2025-01 ~ 2026-12 各月份，便于报告生成场景演示。

用法：python scripts/generate_demo_records.py
"""
import csv
import os
import sys

# 允许以 `python scripts/generate_demo_records.py` 方式直接从工程根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.path_tool import get_abs_path  # noqa: E402

USER_IDS = [f"{i:03d}" for i in range(1, 8)]

# 每个用户的房屋特征，模拟不同户型
HOUSE_FEATURES = {
    "001": "两居室 62㎡ · 全屋清扫",
    "002": "三居室 89㎡ · 全屋清扫 + 拖地",
    "003": "复式 118㎡ · 分区清扫",
    "004": "小户型 45㎡ · 每日定时清扫",
    "005": "三居室 96㎡ · 地毯区域禁扫",
    "006": "两居室 70㎡ · 养宠家庭深度清扫",
    "007": "三居室 105㎡ · 全屋清扫 + 拖地",
}

MONTHS = [f"{year}-{month:02d}" for year in (2025, 2026) for month in range(1, 13)]

CSV_HEADER = ["user_id", "month", "feature", "efficiency", "consumables", "comparison", "time"]


def build_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for user_index, user_id in enumerate(USER_IDS):
        area = 45 + user_index * 10
        previous_rate = None

        for month_index, month in enumerate(MONTHS):
            # 耗材使用月份数：每 6 个月更换一次滤网，形成周期性提醒
            consumable_month = month_index % 6 + 1
            duration = 38 + ((month_index * 3 + user_index * 5) % 22)
            area_covered = area + (month_index % 4) * 2
            rate = round(area_covered / duration * 60, 1)  # 每小时清洁面积

            if previous_rate is None:
                comparison = "首次记录"
            else:
                delta = rate - previous_rate
                trend = "提升" if delta >= 0 else "下降"
                comparison = f"较上月清洁效率{trend} {abs(delta):.1f}%"
            previous_rate = rate

            if consumable_month <= 3:
                consumables = "滤网/边刷状态良好，无需更换"
            elif consumable_month <= 5:
                consumables = "边刷轻微磨损，建议 1 个月内更换"
            else:
                consumables = "滤网接近寿命上限，建议本月更换"

            efficiency = (
                f"覆盖面积 {area_covered}㎡ · 清扫时长 {duration}min · "
                f"清洁效率 {rate}㎡/h"
            )

            rows.append(
                [
                    user_id,
                    month,
                    HOUSE_FEATURES[user_id],
                    efficiency,
                    consumables,
                    comparison,
                    month,
                ]
            )
    return rows


def main() -> None:
    target_path = get_abs_path("data/external/records.csv")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    rows = build_rows()
    with open(target_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    print(f"已生成演示数据：{target_path}，共 {len(rows)} 条记录")


if __name__ == "__main__":
    main()