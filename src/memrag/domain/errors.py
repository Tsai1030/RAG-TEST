"""Domain errors. Messages say which stage and which item failed."""


class MemRAGError(Exception):
    """Base class for all project errors."""


class ConfigError(MemRAGError):
    """A configuration file is missing a field or has an invalid value."""


class ModelNotAvailableError(MemRAGError):
    """A model is not in the registry or has not been downloaded."""


class LLMUnavailableError(MemRAGError):
    """The LLM server did not answer after the network retries."""
