#!/usr/bin/env python3
"""
任务触发判断工具
判断某个任务是否应该触发，支持概率型和间隔型。

用法：
    python check_trigger.py --task publish_whisper --last-run 2026-06-20T21:50:00+08:00

输出（JSON 格式）：
    {
      "task": "publish_whisper",
      "type": "probabilistic",
      "trigger": false,
      "reason": "随机判断未命中",
      "probability": 0.35,
      "actual_probability": 0.35,
      "time_slot": "9-12",
      "day_type": "restday",
      "holiday_name": null,
      "wait_minutes": 0
    }
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone, date

# 确保同目录下的模块可以被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_holiday import load_holidays, check_holiday


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
        # 概率型任务，workday 和 restday 两套
        # 调用方需要先判断是工作日还是休息日
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
    
    # 把 key 转成数字，排序
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
    # 处理带时区的格式
    dt_str = dt_str.strip()
    if dt_str.endswith('Z'):
        dt_str = dt_str[:-1] + '+00:00'
    
    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        # 尝试其他格式
        for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        raise


def check_probabilistic_trigger(task_name, task_config, last_run_dt, now_dt, holidays_data, day_info_param):
    """
    判断概率型任务是否触发
    """
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
    
    # 判断日期类型（在 main 里先算好传进来）
    day_info = day_info_param
    
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
    
    result = {
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
    
    # 如果没触发，计算下次检查时间？
    # 不用，概率型的每次心跳都要判断
    
    return result


def check_interval_trigger(task_name, task_config, last_run_dt, now_dt, random_offset_minutes=None):
    """
    判断间隔型任务是否触发
    """
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
            # 如果没提供，从配置里取范围生成
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


def main():
    parser = argparse.ArgumentParser(description='任务触发判断工具')
    parser.add_argument('--config', default='.whisper-task/config.json', help='配置文件路径')
    parser.add_argument('--task', required=True, help='任务名（如 publish_whisper）')
    parser.add_argument('--last-run', default=None, help='上次执行时间（ISO 格式）')
    parser.add_argument('--now', default=None, help='当前时间（ISO 格式），默认系统时间')
    parser.add_argument('--holidays-file', default='.whisper-task/holidays.json', help='节假日数据文件路径')
    parser.add_argument('--random-seed', type=int, default=None, help='随机种子（用于测试）')
    parser.add_argument('--random-offset', type=int, default=None, help='随机偏移分钟数（间隔型用，用于测试）')
    
    args = parser.parse_args()
    
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
        # 先判断日期类型（复用 check_holiday 脚本的逻辑）
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
    
    # 触发返回 0，不触发返回 1？
    # 还是都返回 0，靠 JSON 里的 trigger 字段判断？
    # 靠 JSON 字段更灵活
    return 0


if __name__ == '__main__':
    exit(main())
