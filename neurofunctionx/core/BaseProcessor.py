from abc import ABC
from datetime import datetime


class BaseProcessor(ABC):
    """
    Base logging functionality
    """
    time_log = None
    verbose = False


    @staticmethod
    def log(message):
        print(f"{datetime.now().strftime('%H:%M:%S')}: " + message)

    def _log(self, message):
        print(f"{datetime.now().strftime('%H:%M:%S')} - {self.__class__.__name__}: " + message)

    def _log_verbose(self, message):
        print(f"{datetime.now().strftime('%H:%M:%S')} - {self.__class__.__name__}: " + message)

    def _log_warn(self, message):
        print(f"{datetime.now().strftime('%H:%M:%S')} - WARN in {self.__class__.__name__}: " + message)

    def _log_error(self, message):
        print(f"{datetime.now().strftime('%H:%M:%S')} - !ERROR! in {self.__class__.__name__}: " + message)

    def _log_time(self, silent=False):
        now = datetime.now()
        if self.time_log is None:
            if not silent:
                print(f"{now.strftime('%H:%M:%S')} - {self.__class__.__name__}: Time log started")
            self.time_log = now
            return None
        else:
            delta = now - self.time_log
            minutes = delta.total_seconds() / 60
            if not silent:
                print(
                    f"{now.strftime('%H:%M:%S')} - {self.__class__.__name__}: "
                    f"Time log stopped. Time Delta: {minutes:.2f} minutes")
            self.time_log = None
            return minutes
