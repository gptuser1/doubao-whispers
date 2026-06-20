#!/usr/bin/env python3
"""
节假日判断工具

用法：
    python check_holiday.py <date> [--holidays-file path/to/holidays.json]

date 格式：YYYY-MM-DD，不填则默认今天

输出（JSON 格式）：
    {
      "date": "2026-06-20",
      "type": "workday",        // workday / weekend / holiday
      "is_workday": true,       // 是否工作日（调休的周末也算）
      "holiday_name": null,     // 节日名称，不是节假日则为 null
      "weekday": "星期六"        // 星期几
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime, date


def load_holidays(holidays_file):
    """加载节假日数据"""
    if not os.path.exists(holidays_file):
        print(f"错误：节假日文件不存在：{holidays_file}", file=sys.stderr)
        return None

    with open(holidays_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_holiday(date_str, holidays_file):
    """
    判断某天是什么类型

    Args:
        date_str: 日期字符串 YYYY-MM-DD
        holidays_file: 节假日数据文件路径

    Returns:
        字典，包含 date, type, is_workday, holiday_name, weekday
    """
    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    year = d.year

    # 加载节假日数据
    holidays_data = load_holidays(holidays_file)
    if holidays_data is None:
        return None

    # 找到对应年份的数据
    year_data = None
    # holidays.json 可能是 { "2026": { holidays: [], workdays: [] } } 格式
    # 也可能是直接的数组，需要看具体格式
    if str(year) in holidays_data:
        year_data = holidays_data[str(year)]
    elif 'holidays' in holidays_data:
        # 可能直接就是当年的数据
        year_data = holidays_data
    else:
        # 尝试找有没有 holidays 数组在顶层
        year_data = holidays_data

    # 提取节假日和调休工作日列表
    holidays_list = []
    workdays_list = []
    holiday_names = {}  # date -> name

    if year_data:
        # 处理 holidays 数组
        raw_holidays = year_data.get('holidays', [])
        for h in raw_holidays:
            if isinstance(h, dict):
                # 格式：{ "name": "端午节", "dates": ["2026-06-19", "2026-06-20", "2026-06-21"] }
                name = h.get('name', '')
                dates = h.get('dates', [])
                for d_str in dates:
                    holidays_list.append(d_str)
                    holiday_names[d_str] = name
            elif isinstance(h, str):
                holidays_list.append(h)

        # 处理 workdays 数组（调休上班日）
        raw_workdays = year_data.get('workdays', [])
        for w in raw_workdays:
            if isinstance(w, str):
                workdays_list.append(w)

    # 判断
    date_str = d.strftime('%Y-%m-%d')
    weekday_cn = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][d.weekday()]

    # 优先判断调休工作日
    if date_str in workdays_list:
        return {
            'date': date_str,
            'type': 'workday',
            'is_workday': True,
            'holiday_name': None,
            'weekday': weekday_cn,
            'note': '调休上班'
        }

    # 再判断节假日
    if date_str in holidays_list:
        return {
            'date': date_str,
            'type': 'holiday',
            'is_workday': False,
            'holiday_name': holiday_names.get(date_str, ''),
            'weekday': weekday_cn
        }

    # 都不是，按星期几判断
    if d.weekday() < 5:  # 周一到周五
        return {
            'date': date_str,
            'type': 'workday',
            'is_workday': True,
            'holiday_name': None,
            'weekday': weekday_cn
        }
    else:  # 周六周日
        return {
            'date': date_str,
            'type': 'weekend',
            'is_workday': False,
            'holiday_name': None,
            'weekday': weekday_cn
        }


def main():
    parser = argparse.ArgumentParser(description='节假日判断工具')
    parser.add_argument('date', nargs='?', default=None, help='日期（YYYY-MM-DD），不填则今天')
    parser.add_argument('--holidays-file', default='.whisper-task/holidays.json', help='节假日数据文件路径')

    args = parser.parse_args()

    # 默认今天
    if not args.date:
        args.date = date.today().strftime('%Y-%m-%d')

    result = check_holiday(args.date, args.holidays_file)

    if result is None:
        return 1

    # 输出 JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == '__main__':
    exit(main())
