import atexit
import threading

TRIGGER = "/mnt/c/Users/jhx/wsl_lock_trigger.txt"
_VERBOSE = True   # 如果想静默运行改为 False

# 模块级状态锁与标志
_lock = threading.Lock()
_active = False

def _log(msg: str):
    """可控日志输出"""
    if _VERBOSE:
        print(f"[stay_awake_wsl] {msg}")

def start_stay_awake():
    """通知 Windows 保持唤醒"""
    global _active
    with _lock:
        if _active:
            return  # 已激活，避免重复写入
        try:
            with open(TRIGGER, "w") as f:
                f.write("1")
            _active = True
            _log("已通知 Windows 保持唤醒")
        except Exception as e:
            _log(f"⚠️ 无法写入 {TRIGGER}: {e}")

def stop_stay_awake():
    """通知 Windows 恢复休眠"""
    global _active
    with _lock:
        if not _active:
            return  # 未激活，无需写入
        try:
            with open(TRIGGER, "w") as f:
                f.write("0")
            _active = False
            _log("已通知 Windows 恢复休眠")
        except Exception as e:
            _log(f"⚠️ 无法写入 {TRIGGER}: {e}")

# 模块被导入时执行一次
start_stay_awake()

# 程序退出时自动恢复
atexit.register(stop_stay_awake)
