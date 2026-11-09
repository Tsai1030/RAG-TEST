"""Domain errors. Messages say which stage and which item failed."""


class MemRAGError(Exception):
    """Base class for all project errors."""


class ConfigError(MemRAGError):
    """A configuration file is missing a field or has an invalid value."""


class ModelNotAvailableError(MemRAGError):
    """A model is not in the registry or has not been downloaded."""


class LLMUnavailableError(MemRAGError):
    """The LLM server did not answer after the network retries."""


class PipelineError(MemRAGError):
    """One pipeline stage failed for one item (a log, a question)."""

    def __init__(self, stage: str, item: str, detail: str):
        super().__init__(f"stage {stage}, {item}: {detail}")
        self.stage, self.item = stage, item
