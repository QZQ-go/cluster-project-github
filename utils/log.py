"""
LOG文件的配置
"""

import logging.handlers
from logging.handlers import WatchedFileHandler, RotatingFileHandler, TimedRotatingFileHandler
import time
import os.path
from stat import ST_DEV, ST_INO
from typing import Optional

try:
    from colorama import Fore, Back
except ModuleNotFoundError as e:
    print('{}-->{}'.format(e, 'log color is could not used'))
except Exception as e:
    print('{}-->{}'.format(e, 'log.py don‘t know what happen'))


class CustomWatchedFileHandler(WatchedFileHandler):

    def __init__(self, filename, mode='a', encoding=None, delay=False, color=True, write=True):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        self.color = color
        self.write = write

    def reopenIfNeeded(self):
        try:
            # stat the file by path, checking for existence
            sres = os.stat(self.baseFilename)
        except FileNotFoundError:
            sres = None
        # compare file system stat with that of our stream file handle
        if not sres or sres[ST_DEV] != self.dev or sres[ST_INO] != self.ino:
            if self.stream is not None:
                # we have an open file handle, clean it up
                self.stream.flush()
                self.stream.close()
                self.stream = None  # See Issue #21742: _open () might fail.
                # open a new file handle and get new stat info from that fd
                if not os.path.exists(os.path.dirname(self.baseFilename)):
                    os.makedirs(os.path.dirname(self.baseFilename))
                self.stream = self._open()
                self._statstream()
                return True

    def emit(self, record: logging.LogRecord) -> None:
        self.reopenIfNeeded()
        if self.write:
            if self.stream is None:
                self.stream = self._open()
            if self.color:
                CustomStreamHandler.emit(self, record)
            else:
                logging.FileHandler.emit(self, record)


def init_custom_fore_color(fore_color, msg: str):
    return fore_color + msg + Fore.RESET


def init_custom_fore_back_color(fore_color, back_color, msg: str):
    return fore_color + back_color + msg + Fore.RESET + Back.RESET


def init_blue_msg(msg: str):
    return init_custom_fore_color(Fore.BLUE, msg)


def init_green_msg(msg: str):
    return init_custom_fore_color(Fore.GREEN, msg)


def init_yellow_msg(msg: str):
    return init_custom_fore_color(Fore.YELLOW, msg)


def init_red_msg(msg: str):
    return init_custom_fore_color(Fore.RED, msg)


class CustomStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno == logging.DEBUG:
                msg = init_green_msg(msg)
            elif record.levelno == logging.INFO:
                msg = init_blue_msg(msg)
            elif record.levelno == logging.WARNING:
                msg = init_yellow_msg(msg)
            elif record.levelno == logging.ERROR:
                msg = init_red_msg(msg)
            elif record.levelno == logging.CRITICAL:
                msg = init_custom_fore_back_color(Fore.RED, Back.WHITE, msg)
            stream = self.stream
            write_msg = msg + self.terminator
            # if 'b' in stream.mode:
            #     stream.write(write_msg.encode(encoding='utf8'))
            # else:
            stream.write(write_msg)
            #     stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False,
                 color=True, watch_file_handler: CustomWatchedFileHandler = None):
        super().__init__(filename, when=when, interval=interval, backupCount=backupCount, encoding=encoding,
                         delay=delay, utc=utc)
        self.color = color
        self.watch_file_handler = watch_file_handler

    def emit(self, record: logging.LogRecord) -> None:
        if self.watch_file_handler:
            res = self.watch_file_handler.reopenIfNeeded()
            if res:
                self.stream.flush()
                self.stream.close()
                self.stream = None
        try:
            if self.shouldRollover(record):
                self.watch_file_handler.stream.close()
                self.watch_file_handler.stream = None
                self.doRollover()
                self.watch_file_handler.stream = self.watch_file_handler._open()
            if self.stream is None:
                self.stream = self._open()
            if self.color:
                CustomStreamHandler.emit(self, record)
            else:
                logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)


class CustomFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = ...) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            t = time.strftime("%Y-%m-%dT%H:%M:%S%z", ct)
            dt, zone = t.rsplit('+', 1)
            s = dt + ".%03d+" % record.msecs + zone
            # s = str(datetime.datetime.now())
        return s


default_formatter = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s - %(lineno)d - %(message)s'
default_suffix = '%Y-%m-%d_%H-%M-%S.my_log'


# date_format = '%Y-%m-%dT%H-%M-%S.%f'


def __init_common_file_handler(file_name, file_handler_class, level=logging.INFO, formatter=default_formatter,
                               suffix: str = default_suffix, **kwargs):
    kwargs['filename'] = file_name
    if not os.path.exists(os.path.dirname(os.path.abspath(file_name))):
        os.makedirs(os.path.dirname(file_name))
    file_handler = file_handler_class(**kwargs)
    # 设置后缀名称，跟strf_time的格式一样
    file_handler.suffix = suffix
    file_handler.setLevel(level)
    formatter = CustomFormatter(formatter)
    # formatter = logging.Formatter(formatter, date_fmt=date_format)
    file_handler.setFormatter(formatter)
    return file_handler


def __init_time_rotating_file_handler(file_name, level=logging.INFO, formatter=default_formatter, when: str = 'D',
                                      interval: int = 3, backup_count: int = 5, suffix: str = default_suffix,
                                      color: bool = True, watch_file_handler: CustomWatchedFileHandler = None):
    kwargs = {'backupCount': backup_count, 'encoding': 'utf-8', 'when': when, 'interval': interval, 'color': color,
              'watch_file_handler': watch_file_handler}
    return __init_common_file_handler(file_name, CustomTimedRotatingFileHandler, level=level, formatter=formatter,
                                      suffix=suffix, **kwargs)


def __init_watched_file_handler(file_name, level=logging.INFO, formatter=default_formatter,
                                suffix: str = default_suffix):
    kwargs = {'encoding': 'utf-8'}
    return __init_common_file_handler(file_name, WatchedFileHandler, level=level, formatter=formatter
                                      , suffix=suffix, **kwargs)


def __init_custom_watched_file_handler(file_name, level=logging.INFO, formatter=default_formatter,
                                       suffix: str = default_suffix, color: bool = True, write: bool = True):
    kwargs = {'encoding': 'utf-8', 'mode': 'a', 'color': color, 'write': write}
    return __init_common_file_handler(file_name, CustomWatchedFileHandler, level=level, formatter=formatter
                                      , suffix=suffix, **kwargs)


def __init_rotating_file_handler(file_name, level=logging.INFO, formatter=default_formatter,
                                 suffix: str = default_suffix, max_bytes: int = 10 * 1024 * 1024,
                                 backup_count: int = 5):
    kwargs = {'encoding': 'utf-8', 'backupCount': backup_count, 'maxBytes': max_bytes}
    return __init_common_file_handler(file_name, RotatingFileHandler, level=level, formatter=formatter,
                                      suffix=suffix, **kwargs)


def __init_console_handler(level=logging.DEBUG, formatter=default_formatter):
    console_handler = CustomStreamHandler()
    console_handler.setLevel(level)
    formatter = CustomFormatter(formatter)
    # formatter = logging.Formatter(formatter, datefmt=date_format)
    console_handler.setFormatter(formatter)
    return console_handler


def init_console_and_file_log(log_name: str, log_file_path: str, color=False, when='D', interval=3) -> logging.Logger:
    """
    初始化带有控制台以及文件的log 时间-名字-级别-信息
    :param interval:
    :param when:
    :param color: 是否在日志里添加颜色
    :param log_name: 日志名字
    :param log_file_path: 日志文件所在路径
    :return:
    """
    # 初始化错误处理日志
    _log = logging.getLogger(log_name)
    if len(_log.handlers) > 0:
        return _log
    _log.setLevel(logging.DEBUG)
    watched_file_handler = __init_custom_watched_file_handler(log_file_path, color=color, write=False)
    time_rotating_file_handler = __init_time_rotating_file_handler(
        log_file_path, color=color, when=when, interval=interval, watch_file_handler=watched_file_handler)
    console_handler = __init_console_handler()
    _log.addHandler(console_handler)
    # _log.addHandler(watched_file_handler)
    _log.addHandler(time_rotating_file_handler)
    return _log


def init_console_log(log_name: str) -> logging.Logger:
    """
    初始化带有控制台的log 时间-名字-级别-信息
    :param log_name:
    :return:
    """
    _log = logging.getLogger(log_name)
    _log.setLevel(logging.DEBUG)
    console_handler = __init_console_handler()
    _log.addHandler(console_handler)
    return _log
