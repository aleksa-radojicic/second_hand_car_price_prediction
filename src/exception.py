class IPAddressBlockedError(RuntimeError):
    def __init__(self, msg):
        super().__init__(msg)

class ScrapingError(RuntimeError):
    def __init__(self, msg):
        super().__init__(msg)

class LabelNotGivenException(RuntimeError):
    def __init__(self, msg):
        super().__init__(msg)
