import sys
sys.path.insert(0, '../')
import logging
import os

def setup_logging(log_path=None):
    """
    配置全局日志格式
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 终端输出 Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        if log_path:
            if os.path.isdir(log_path) or not log_path.endswith(('.log', '.txt')):
                os.makedirs(log_path, exist_ok=True)  # 确保文件夹存在
                actual_log_file = os.path.join(log_path, 'inference.log') # 自动补上文件名
            else:
                parent_dir = os.path.dirname(log_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                actual_log_file = log_path

            file_handler = logging.FileHandler(actual_log_file, encoding='utf-8')
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