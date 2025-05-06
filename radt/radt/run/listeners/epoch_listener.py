import sys
import re
import tempfile
import logging

class EpochListener:
    """
    EpochListener fanger stdout og stderr for at detektere epoch-skift via
    foruddefinerede regex patterns.
    """
    def __init__(self, callback, pattern=None):
        self.logger = logging.getLogger("EpochListener")
        self.callback = callback
        if isinstance(pattern, str):
            self.pattern = [re.compile(pattern)]

        print(f"PATTERN: {self.pattern}")
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.capture_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
        self.capture_filename = self.capture_file.name
        self.logger.info(f"Created temporary log capture file: {self.capture_filename}")
        self.is_capturing = False

    def start_capture(self):
        """
        Omdirigerer stdout og stderr til en output capture, som tjekker for epoch-skift.
        """
        if self.is_capturing:
            return
        self.is_capturing = True

        class OutputCapture:
            def __init__(self, original, capture_file, callback, pattern):
                self.original = original
                self.capture_file = capture_file
                self.callback = callback
                self.pattern = pattern
                self.buffer = ""
                
            def write(self, text):
                self.original.write(text)
                self.capture_file.write(text)
                self.capture_file.flush()
                self.buffer += text
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines.pop()
                    for line in lines:
                        for p in self.pattern:
                            match = re.search(p, line)
                            if match:
                                try:
                                    epoch = int(match.group(1))
                                    self.callback(epoch, line)
                                    break
                                except Exception:
                                    continue
            def flush(self):
                self.original.flush()
                self.capture_file.flush()

        sys.stdout = OutputCapture(self.original_stdout, self.capture_file, self.callback, self.pattern)
        sys.stderr = OutputCapture(self.original_stderr, self.capture_file, self.callback, self.pattern)
        self.logger.info("Stdout/stderr redirection set up for epoch detection")

    def stop_capture(self):
        """
        Gendanner original stdout/stderr og lukker den midlertidige fil.
        """
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if not self.capture_file.closed:
            self.capture_file.close()
        self.logger.info("Restored original stdout/stderr and closed capture file")
