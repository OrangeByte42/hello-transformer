from typing import Tuple


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds
    @param start_time: Start time in seconds
    @param end_time: End time in seconds
    @return: Tuple of elapsed time in minutes and seconds
    """
    elapsed_time: float = end_time - start_time
    elapsed_mins: int = int(elapsed_time // 60)
    elapsed_secs: int = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs




