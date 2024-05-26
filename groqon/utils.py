from datetime import datetime


def get_current_time_str():
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return time_str

