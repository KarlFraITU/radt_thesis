import os
import re
import glob
import mlflow
import threading
import logging
import tempfile
import sys
from time import sleep, time
from carbontracker.tracker import CarbonTracker

class CarbonTrackerListener(threading.Thread):
    """
    Enhanced CarbonTracker that captures training output to detect epoch changes
    and properly synchronize carbon tracking with the training process.
    """
    
    def __init__(self, run_id, experiment_id=88, carbon_log_dir='./carbon_logs', epoch_duration_threshold=20,  epoch_patterns=None):
        """
        Initialize the enhanced carbon tracker.
        
        Args:
            run_id: Existing MLflow run ID
            experiment_id: MLflow experiment ID (optional)
            carbon_log_dir: Directory where carbon tracker logs will be stored
            epoch_duration_threshold: Minimum duration (in seconds) for an epoch before forcing a new one
        """
        super().__init__(daemon=True)
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EnhancedCarbonTracker")
        
        self.carbon_log_dir = carbon_log_dir
        self.max_epochs = int(os.getenv("RADT_MAX_EPOCH", 5))
        self.epoch_duration_threshold = epoch_duration_threshold

        # Use passed epoch_patterns if provided; otherwise, use default patterns
        self.epoch_patterns = epoch_patterns if epoch_patterns is not None else [
            r'Epoch\s*(\d+)(?:/\d+)?',
            r'(?:Training|Epoch)[:\s=]+(\d+)',
            r'Epoch\s+(\d+)/\d+',
            r'Epoch\s*#?\s*(\d+)',
            r'epoch[:\s=]+(\d+)',
            r'EPOCH\s*(\d+)',
            r'^\[(\d+),'
        ]
        
        # Create log directory if it doesn't exist
        os.makedirs(carbon_log_dir, exist_ok=True)
        
        # MLflow run management
        if run_id:
            self.run_id = run_id
            self.should_manage_run = False
            self.logger.info(f"Using existing MLflow run: {self.run_id}")
        elif mlflow.active_run():
            self.run_id = mlflow.active_run().info.run_id
            self.should_manage_run = False
            self.logger.info(f"Using active MLflow run: {self.run_id}")
        else:
            run = mlflow.start_run(experiment_id=experiment_id)
            self.run_id = run.info.run_id
            self.should_manage_run = True
            self.logger.info(f"Started new MLflow run: {self.run_id}")

        # Prepare stdout capturing
        self.capture_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
        self.capture_filename = self.capture_file.name
        self.logger.info(f"Created temporary log capture file: {self.capture_filename}")
        
        # Internal state tracking
        self.current_epoch = -1  # We'll update this on the first detected (or forced) epoch
        self.last_epoch_time = time()
        self.latest_log_file = None  # Store the latest log file path
        self._stop_event = threading.Event()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Initialize the CarbonTracker instance
        self.carbon_tracker = CarbonTracker(
            epochs=self.max_epochs,
            log_dir=carbon_log_dir,
            update_interval=10  # More frequent updates to catch short epochs
        )
        self.logger.info(f"Initialized CarbonTracker with max epochs: {self.max_epochs}")

    def setup_output_capture(self):
        """
        Set up stdout/stderr redirection to capture all output from the training script.
        This is key for detecting epoch changes without modifying the training script.
        """
        class OutputCapture:
            def __init__(self, original, capture_file, callback, patterns):
                self.original = original
                self.capture_file = capture_file
                self.callback = callback
                self.patterns = patterns
                self.buffer = ""
                
            def write(self, text):
                self.original.write(text)
                self.capture_file.write(text)
                self.capture_file.flush()
                self.buffer += text
                if '\n' in text:
                    lines = self.buffer.split('\n')
                    self.buffer = lines.pop()  # keep the last incomplete line
                    for line in lines:
                        for pattern in self.patterns:
                            match = re.search(pattern, line)
                            if match:
                                try:
                                    epoch = int(match.group(1))
                                    self.callback(epoch, line)
                                    break
                                except (ValueError, IndexError):
                                    continue
            def flush(self):
                self.original.flush()
                self.capture_file.flush()
        
        sys.stdout = OutputCapture(self.original_stdout, self.capture_file, self._on_epoch_detected, self.epoch_patterns)
        sys.stderr = OutputCapture(self.original_stderr, self.capture_file, self._on_epoch_detected, self.epoch_patterns)
        self.logger.info("Stdout/stderr redirection set up for epoch detection")

    def restore_output(self):
        """Restore original stdout/stderr"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if not self.capture_file.closed:
            self.capture_file.close()
        self.logger.info("Restored original stdout/stderr")

    def _on_epoch_detected(self, epoch, line):
        """
        Called when a new epoch is detected from the output.
        
        Args:
            epoch: The detected epoch number.
            line: The full line containing the epoch information.
        """
        if epoch != self.current_epoch:
            self.logger.info(f"Detected new epoch: {epoch} in line: {line.strip()}")
            if self.current_epoch >= 0:
                try:
                    self.carbon_tracker.epoch_end()
                    self.parse_and_log_metrics(self.current_epoch)
                except Exception as e:
                    self.logger.error(f"Error ending epoch {self.current_epoch}: {e}")
            self.current_epoch = epoch
            self.logger.info(f"Updated epoch counter to: {self.current_epoch}")
            self.last_epoch_time = time()
            try:
                self.carbon_tracker.epoch_start()
            except Exception as e:
                self.logger.error(f"Error starting epoch {epoch}: {e}")

    def parse_and_log_metrics(self, epoch):
        """Parse CarbonTracker logs and log metrics to MLflow."""
        log_files = glob.glob(os.path.join(self.carbon_log_dir, '*.log'))
        if not log_files:
            self.logger.warning("No CarbonTracker log files found yet.")
            return

        self.latest_log_file = max(log_files, key=os.path.getctime)
        self.logger.info(f"Processing log file: {self.latest_log_file}")
        try:
            with open(self.latest_log_file, 'r') as f:
                log_content = f.read()

            # Find all GPU power usage entries
            power_matches = re.findall(
                r'Epoch\s*(\d+):.*?Average power usage \(W\) for gpu:\s*(\d+\.?\d*)',
                log_content,
                re.DOTALL | re.IGNORECASE
            )
            # Find overall carbon intensity
            carbon_intensity_match = re.search(
                r'Average carbon intensity during training was (\d+\.?\d*) gCO2/kWh',
                log_content,
                re.IGNORECASE
            )
            
            metrics = {}
            for matched_epoch, gpu_power in power_matches:
                try:
                    if int(matched_epoch) == epoch:
                        metrics['Carbontracker - GPU Power W'] = float(gpu_power)
                        self.logger.info(f"Epoch {epoch}: GPU Power = {gpu_power}W")
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not parse epoch or power from match: {matched_epoch}, {gpu_power}")

            if carbon_intensity_match:
                try:
                    carbon_intensity = float(carbon_intensity_match.group(1))
                    metrics['Carbontracker - Carbon Intensity gCO2/kWh'] = carbon_intensity
                    self.logger.info(f"Carbon Intensity: {carbon_intensity} gCO2/kWh")
                except (ValueError, IndexError):
                    self.logger.warning(f"Could not parse carbon intensity from match: {carbon_intensity_match.group(0)}")
            
            if metrics:
                self.logger.info(f"Logging metrics for epoch {epoch}: {metrics}")
                mlflow.log_metrics(metrics, step=epoch)
            else:
                self.logger.warning(f"No metrics found to log for epoch {epoch}")
                
        except Exception as e:
            self.logger.error(f"Error parsing or logging metrics from file {self.latest_log_file}: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        self.logger.info("Starting EnhancedCarbonTracker")
        self.setup_output_capture()
        mlflow.start_run(run_id=self.run_id).__enter__()
        self.carbon_tracker.epoch_start()
        self.current_epoch = 0
        self.last_epoch_time = time()

        try:
            while not self._stop_event.is_set() and self.current_epoch < self.max_epochs:
                time_now = time()
                time_since_last_epoch = time_now - self.last_epoch_time

                if time_since_last_epoch >= self.epoch_duration_threshold:
                    # Enhanced fallback mechanism: Only force epoch if it has genuinely taken too long
                    forced_epoch = self.current_epoch + 1
                    self.logger.info(f"Fallback triggered: Forcing new epoch: {forced_epoch}")
                    self._on_epoch_detected(forced_epoch, f"Fallback forced epoch {forced_epoch}")
                    self.logger.info(f"Epoch counter is now {self.current_epoch}")
                
                # Shorter sleep for more responsiveness
                sleep(1)  # Reduced from 5 seconds to 1 second for faster epoch checks
        except Exception as e:
            self.logger.error(f"Error in EnhancedCarbonTracker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure final measurements are logged
            try:
                self.carbon_tracker.epoch_end()
            except Exception as e:
                self.logger.error(f"Error ending final epoch: {e}")
            if self.latest_log_file and os.path.exists(self.latest_log_file):
                try:
                    mlflow.log_artifact(self.latest_log_file, artifact_path='carbon_logs')
                except Exception as e:
                    self.logger.error(f"Error logging final carbon log artifact: {e}")
            self.carbon_tracker.stop()
            mlflow.end_run()
            self.restore_output()

    def terminate(self):
        """Signal the thread to terminate."""
        self.logger.info("Terminating EnhancedCarbonTracker")
        self._stop_event.set()
        try:
            self.carbon_tracker.epoch_end()
        except Exception:
            pass
        self.carbon_tracker.stop()
