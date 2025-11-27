import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    _is_configured = False

    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

        if not CustomLogger._is_configured:
            self._configure()
            CustomLogger._is_configured = True

    def _configure(self):
        """Configure logging + structlog once globally."""
        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
        console_handler = logging.StreamHandler()

        handlers = [console_handler, file_handler]

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # structlog already produces JSON
            handlers=handlers,
            force=True,  # Python 3.8+: ensures config overrides defaults
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )

    def get_logger(self, name: str):
        """Return a structlog logger with a clean name."""
        logger_name = os.path.basename(name).replace(".py", "")
        return structlog.get_logger(logger_name)
