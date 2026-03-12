import sys
sys.path.insert(0, '../')
import logging

def setup_logging(log_file=None):
    """
    配置全局日志格式
    """
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 统一的格式：时间 - 日志级别 - [进程名] - 消息内容
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 终端输出 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # (可选) 如果需要保存到文件
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def log_message(message, log_file_path):
    print(message)
    try:
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file: {e}")