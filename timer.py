import time

class TimerError(Exception):
  """A custom exception used to report errors in use of Timer class"""

class Timer:
  def __init__(self, id: int):
    self._start_time = None
    self._id = id

  def start(self):
    """Start a new timer"""
    if self._start_time is not None:
      raise TimerError(f"Timer {self._id} is running. Use .stop() to stop it")

    self._start_time = time.perf_counter()

  def stop(self):
    """Stop the timer, and report the elapsed time"""
    if self._start_time is None:
      raise TimerError(f"Timer {self._id} is not running. Use .start() to start it")

    elapsed_time = time.perf_counter() - self._start_time
    self._start_time = None
    return elapsed_time
    # print(f"Timer {self._id} elapsed time: {elapsed_time:0.4f} seconds")