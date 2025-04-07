import os
import re
import glob
import mlflow
import threading
import sys
import tempfile
from time import sleep, time
from carbontracker.tracker import CarbonTracker
from .epoch_listener import EpochListener

class CarbonTrackerListener(threading.Thread):
    """
    CarbonTrackerListener handles the CarbonTracker instance and coordinates
    with the EpochListener to detect epoch transitions.
    """
    def __init__(self, run_id, experiment_id=88, carbon_log_dir='./carbon_logs'):
        super(CarbonTrackerListener, self).__init__()
        self.run_id = run_id
        self.experiment_id = experiment_id
        
        self.carbon_log_dir = carbon_log_dir
        self.max_epochs = int(os.getenv("RADT_MAX_EPOCH"))

        os.makedirs(carbon_log_dir, exist_ok=True)

        self.current_epoch = -1
        self.last_epoch_time = time()
        self.latest_log_file = None

        # Initialize the CarbonTracker instance
        self.carbon_tracker = CarbonTracker(
            epochs=self.max_epochs,
            log_dir=carbon_log_dir,
            update_interval=10
        )

        # Set up the EpochListener
        self.epoch_listener = EpochListener(callback=self._on_epoch_detected)
        self._stop_event = threading.Event()

    def _on_epoch_detected(self, epoch, line):
        """
        Callback that is called when a new epoch is detected.
        """
        if epoch != self.current_epoch:
            if self.current_epoch >= 0:
                try:
                    self.carbon_tracker.epoch_end()
                    self.parse_and_log_metrics(self.current_epoch)
                except Exception as e:
                    print(f"Error ending epoch {self.current_epoch}: {e}")
            self.current_epoch = epoch
            self.last_epoch_time = time()
            try:
                self.carbon_tracker.epoch_start()
            except Exception as e:
                print(f"Error starting epoch {epoch}: {e}")

    def parse_and_log_metrics(self, epoch):
        """
        Parses CarbonTracker log files and logs metrics to MLflow.
        """
        log_files = glob.glob(os.path.join(self.carbon_log_dir, '*.log'))
        if not log_files:
            print("No CarbonTracker log files found yet.")
            return

        self.latest_log_file = max(log_files, key=os.path.getctime)
        try:
            with open(self.latest_log_file, 'r') as f:
                log_content = f.read()

            power_matches = re.findall(
                r'Epoch\s*(\d+):.*?Average power usage \(W\) for gpu:\s*(\d+\.?\d*)',
                log_content,
                re.DOTALL | re.IGNORECASE
            )
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
                except (ValueError, IndexError):
                    print(f"Could not parse epoch or power from match: {matched_epoch}, {gpu_power}")

            if carbon_intensity_match:
                try:
                    carbon_intensity = float(carbon_intensity_match.group(1))
                    metrics['Carbontracker - Carbon Intensity gCO2/kWh'] = carbon_intensity
                except (ValueError, IndexError):
                    print("Could not parse carbon intensity from log.")
            
            if metrics:
                mlflow.log_metrics(metrics, step=epoch)
                
        except Exception as e:
            print(f"Error parsing or logging metrics from file {self.latest_log_file}: {e}")

    def run(self):
        self.epoch_listener.start_capture()  # Starts stdout/stderr redirection
        mlflow.start_run(run_id=self.run_id).__enter__()
        self.carbon_tracker.epoch_start()
        self.current_epoch = 0
        self.last_epoch_time = time()

        try:
            while not self._stop_event.is_set() and self.current_epoch < self.max_epochs:
                time_now = time()
                sleep(1)
        except Exception as e:
            print(f"Error in CarbonTrackerManager: {e}")
        finally:
            try:
                self.carbon_tracker.epoch_end()
            except Exception as e:
                print(f"Error ending final epoch: {e}")
            if self.latest_log_file and os.path.exists(self.latest_log_file):
                try:
                    mlflow.log_artifact(self.latest_log_file, artifact_path='carbon_logs')
                except Exception as e:
                    print(f"Error logging final carbon log artifact: {e}")
            self.carbon_tracker.stop()
            mlflow.end_run()
            self.epoch_listener.stop_capture()

    def terminate(self):
        self._stop_event.set()
        try:
            self.carbon_tracker.epoch_end()
        except Exception:
            pass
        self.carbon_tracker.stop
