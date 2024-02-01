from src.logger import SeverityMode


class SeverityException(Exception):
    def __init__(self, *args, severity: SeverityMode = SeverityMode.INFO):
        self.severity = severity
        super().__init__(*args)


class IPAddressBlockedException(SeverityException):
    def __init__(self, *args):
        super().__init__(*args)


class ScrapingException(SeverityException):
    def __init__(self, *args):
        super().__init__(*args)


class LabelNotGivenException(SeverityException):
    def __init__(self, *args):
        super().__init__(*args)
