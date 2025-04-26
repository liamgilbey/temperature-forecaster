import logging

class Logger:
    def __init__(self, level=logging.INFO):
        self.logger = logging.getLogger("TemperatureForecasterLogger")
        self.logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)