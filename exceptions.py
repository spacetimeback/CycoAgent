class LongerThanContextError(Exception):
    """
    Exception raised when the input text is longer than the model's context window.
    """
    pass 