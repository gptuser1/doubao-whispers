#!/usr/bin/env python3
"""
任务触发判断工具
整合了节假日判断和任务触发判断两种功能。

用法：
    # 判断任务是否触发（默认功能）
    python check_trigger.py trigger --task publish_whisper --last-run 2026-06-20T21:50:00+08:00
    
    # 只判断节假日
    python check_trigger.py holiday 2026-06-20
    
    # 不传子命令默认是 trigger 模式
    python check_trigger.py --task publish_whisper --last-run ...
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone, date


# ==================== 节假日判断 ====================

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
        # 节假日文件不存在，按星期几判断
        weekday_cn = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][d.weekday()]
        is_workday = d.weekday() < 5
        return {
            'date': date_str,
            'type': 'workday' if is_workday else 'weekend',
            'is_workday': is_workday,
            'holiday_name': None,
            'weekday': weekday_cn,
            'note': '无节假日数据，按星期几判断'
        }

    # 找到对应年份的数据
    year_data = None
    if str(year) in holidays_data:
        year_data = holidays_data[str(year)]
    elif 'holidays' in holidays_data:
        year_data = holidays_data
    else:
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


# ==================== 任务触发判断 ====================

def load_config(config_file):
    """加载配置文件"""
    if not os.path.exists(config_file):
        print(f"错误：配置文件不存在：{config_file}", file=sys.stderr)
        return None
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_time_slot(hour, schedule_config):
    """
    根据小时数找到对应的时间段
    时间段格式："7-9" 表示 7:00-9:00
    """
    # 先看用哪套概率表
    if 'workday' in schedule_config and 'restday' in schedule_config:
        return None, schedule_config
    
    # 直接遍历时间段
    for slot in schedule_config.keys():
        if '-' in slot:
            start, end = slot.split('-')
            start_h = int(start)
            end_h = int(end)
            if start_h <= hour < end_h:
                return slot, schedule_config[slot]
    
    return None, None


def calculate_holiday_duration(holiday_name, holidays_data, target_date):
    """
    计算某个假期有多少天
    用于确定节假日倍率
    """
    if not holidays_data or not holiday_name:
        return 0
    
    year = target_date.year
    year_data = None
    if str(year) in holidays_data:
        year_data = holidays_data[str(year)]
    elif 'holidays' in holidays_data:
        year_data = holidays_data
    
    if not year_data:
        return 0
    
    raw_holidays = year_data.get('holidays', [])
    for h in raw_holidays:
        if isinstance(h, dict) and h.get('name') == holiday_name:
            return len(h.get('dates', []))
    
    return 0


def get_holiday_multiplier(days, multipliers_config):
    """
    根据假期天数获取倍率
    找小于等于实际天数的最大 key
    """
    if not multipliers_config:
        return 1.0
    
    days_list = sorted([int(k) for k in multipliers_config.keys()])
    
    result = 1.0
    for d in days_list:
        if d <= days:
            result = float(multipliers_config[str(d)])
        else:
            break
    
    return result


def parse_iso_datetime(dt_str):
    """解析 ISO 格式的时间字符串"""
    dt_str = dt_str.strip()
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        raise


def check_probabilistic_trigger(task_name, task_config, last_run_dt, now_dt, holidays_data, day_info):
    """判断概率型任务是否触发"""
    schedule = task_config.get('schedule', {})
    min_interval_hours = schedule.get('min_interval_hours', 24)
    
    # 检查最小间隔
    if last_run_dt:
        elapsed = now_dt - last_run_dt
        elapsed_hours = elapsed.total_seconds() / 3600
        
        if elapsed_hours < min_interval_hours:
            wait_minutes = (min_interval_hours - elapsed_hours) * 60
            return {
                'task': task_name,
                'type': 'probabilistic',
                'trigger': False,
                'reason': f'未到最小间隔，还需等待 {wait_minutes:.0f} 分钟',
                'probability': 0,
                'actual_probability': 0,
                'wait_minutes': wait_minutes
            }
    
    # 确定用哪套概率表
    if day_info['is_workday']:
        day_type = 'workday'
    else:
        day_type = 'restday'  # 周末和节假日都用 restday 的基础概率
    
    prob_table = schedule.get(day_type, {})
    if not prob_table:
        return {
            'task': task_name,
            'type': 'probabilistic',
            'trigger': False,
            'reason': f'找不到 {day_type} 的概率配置',
            'probability': 0,
            'actual_probability': 0,
            'day_type': day_type,
            'wait_minutes': 0
        }
    
    # 找到当前时间段的概率
    hour = now_dt.hour
    time_slot, probability = get_time_slot(hour, prob_table)
    
    if time_slot is None:
        return {
            'task': task_name,
            'type': 'probabilistic',
            'trigger': False,
            'reason': f'当前时间 {hour}:00 不在任何时间段内',
            'probability': 0,
            'actual_probability': 0,
            'day_type': day_type,
            'wait_minutes': 0
        }
    
    actual_probability = probability
    
    # 如果是节假日，应用倍率
    is_holiday = day_info['type'] == 'holiday'
    if is_holiday:
        multipliers = schedule.get('holiday_multipliers', {})
        if multipliers:
            holiday_days = calculate_holiday_duration(
                day_info['holiday_name'], 
                holidays_data, 
                now_dt.date()
            )
            multiplier = get_holiday_multiplier(holiday_days, multipliers)
            actual_probability = probability * multiplier
    
    # 生成随机数判断
    random_value = random.random()
    trigger = random_value < actual_probability
    
    reason = "随机判断命中" if trigger else "随机判断未命中"
    
    return {
        'task': task_name,
        'type': 'probabilistic',
        'trigger': trigger,
        'reason': reason,
        'probability': probability,
        'actual_probability': round(actual_probability, 4),
        'random_value': round(random_value, 4),
        'time_slot': time_slot,
        'day_type': day_type,
        'holiday_name': day_info.get('holiday_name'),
        'is_holiday': is_holiday,
        'wait_minutes': 0
    }


def check_interval_trigger(task_name, task_config, last_run_dt, now_dt, random_offset_minutes=None):
    """判断间隔型任务是否触发"""
    schedule = task_config.get('schedule', {})
    min_interval_hours = schedule.get('min_interval_hours', 1)
    
    # 时间段限制
    only_between = schedule.get('only_between_hours', [])
    
    if only_between and len(only_between) == 2:
        start_h, end_h = only_between
        if now_dt.hour < start_h or now_dt.hour >= end_h:
            return {
                'task': task_name,
                'type': 'interval',
                'trigger': False,
                'reason': f'不在允许时间段内（{start_h}:00-{end_h}:00）',
                'wait_minutes': 0
            }
    
    # 检查最小间隔
    if last_run_dt:
        # 随机偏移
        if random_offset_minutes is None:
            offset_min = schedule.get('random_offset_min_minutes', 0)
            offset_max = schedule.get('random_offset_max_minutes', 0)
            if offset_min < offset_max:
                random_offset_minutes = random.randint(offset_min, offset_max)
            else:
                random_offset_minutes = 0
        
        total_interval_minutes = min_interval_hours * 60 + random_offset_minutes
        
        elapsed = now_dt - last_run_dt
        elapsed_minutes = elapsed.total_seconds() / 60
        
        if elapsed_minutes < total_interval_minutes:
            wait_minutes = total_interval_minutes - elapsed_minutes
            return {
                'task': task_name,
                'type': 'interval',
                'trigger': False,
                'reason': f'未到最小间隔，还需等待 {wait_minutes:.0f} 分钟',
                'wait_minutes': round(wait_minutes, 1),
                'random_offset_minutes': random_offset_minutes
            }
    
    return {
        'task': task_name,
        'type': 'interval',
        'trigger': True,
        'reason': '达到触发条件',
        'wait_minutes': 0,
        'random_offset_minutes': random_offset_minutes
    }


# ==================== 主入口 ====================

def cmd_holiday(args):
    """holiday 子命令：只判断节假日"""
    result = check_holiday(args.date, args.holidays_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_trigger(args):
    """trigger 子命令：判断任务是否触发"""
    # 设置随机种子
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # 加载配置
    config = load_config(args.config)
    if config is None:
        return 1
    
    # 找到任务配置
    operations = config.get('operations', {})
    if args.task not in operations:
        print(f"错误：找不到任务 {args.task}", file=sys.stderr)
        return 1
    
    task_config = operations[args.task]
    
    # 检查是否启用
    if not task_config.get('enabled', True):
        result = {
            'task': args.task,
            'type': task_config.get('schedule', {}).get('type', 'unknown'),
            'trigger': False,
            'reason': '任务已禁用'
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    
    # 加载节假日数据
    holidays_data = load_holidays(args.holidays_file)
    
    # 解析时间
    tz = timezone(timedelta(hours=8))  # 北京时间
    
    if args.now:
        now_dt = parse_iso_datetime(args.now)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=tz)
    else:
        now_dt = datetime.now(tz)
    
    last_run_dt = None
    if args.last_run:
        last_run_dt = parse_iso_datetime(args.last_run)
        if last_run_dt.tzinfo is None:
            last_run_dt = last_run_dt.replace(tzinfo=tz)
    
    # 判断任务类型
    schedule_type = task_config.get('schedule', {}).get('type', 'interval')
    
    if schedule_type == 'probabilistic':
        # 先判断日期类型
        date_str = now_dt.strftime('%Y-%m-%d')
        day_info = check_holiday(date_str, args.holidays_file)
        
        result = check_probabilistic_trigger(
            args.task, task_config, last_run_dt, now_dt, holidays_data, day_info
        )
    elif schedule_type == 'interval':
        result = check_interval_trigger(
            args.task, task_config, last_run_dt, now_dt, args.random_offset
        )
    else:
        result = {
            'task': args.task,
            'type': schedule_type,
            'trigger': False,
            'reason': f'不支持的任务类型：{schedule_type}'
        }
    
    # 输出结果
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    # 处理默认子命令：如果没有指定子命令，默认是 trigger
    # 遍历参数，找第一个非选项参数（不以 - 开头）
    subcommands = {'holiday', 'trigger'}
    has_subcommand = False
    
    for arg in sys.argv[1:]:
        if not arg.startswith('-'):
            if arg in subcommands:
                has_subcommand = True
            break
    
    if not has_subcommand:
        # 没有子命令，默认插入 trigger
        sys.argv.insert(1, 'trigger')
    
    parser = argparse.ArgumentParser(description='任务触发判断工具')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # holiday 子命令
    holiday_parser = subparsers.add_parser('holiday', help='判断节假日')
    holiday_parser.add_argument('date', nargs='?', default=None, help='日期（YYYY-MM-DD），不填则今天')
    holiday_parser.add_argument('--holidays-file', default='.whisper-task/holidays.json', help='节假日数据文件路径')
    holiday_parser.set_defaults(func=cmd_holiday)
    
    # trigger 子命令
    trigger_parser = subparsers.add_parser('trigger', help='判断任务是否触发（默认）')
    trigger_parser.add_argument('--config', default='.whisper-task/config.json', help='配置文件路径')
    trigger_parser.add_argument('--task', required=True, help='任务名（如 publish_whisper）')
    trigger_parser.add_argument('--last-run', default=None, help='上次执行时间（ISO 格式）')
    trigger_parser.add_argument('--now', default=None, help='当前时间（ISO 格式），默认系统时间')
    trigger_parser.add_argument('--holidays-file', default='.whisper-task/holidays.json', help='节假日数据文件路径')
    trigger_parser.add_argument('--random-seed', type=int, default=None, help='随机种子（用于测试）')
    trigger_parser.add_argument('--random-offset', type=int, default=None, help='随机偏移分钟数（间隔型用，用于测试）')
    trigger_parser.set_defaults(func=cmd_trigger)
    
    args = parser.parse_args()
    
    # 如果没有指定子命令，默认按 trigger 处理（向后兼容）
    if args.command is None:
        # 重新解析，把所有参数都当 trigger 的参数
        # 简单做法：直接调用 trigger 解析器
        # 但这样会丢失 --help 等功能
        # 更好的做法：手动检查
        # 这里用简单方式：直接用 trigger 解析器重新解析
        trigger_args = trigger_parser.parse_args(sys.argv[1:])
        return cmd_trigger(trigger_args)
    
    # 执行对应子命令
    return args.func(args)


if __name__ == '__main__':
    exit(main())
