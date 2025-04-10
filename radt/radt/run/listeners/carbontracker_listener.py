import os
import re
import glob
import mlflow
import threading
import sys
import tempfile
from time import sleep, time
from carbontracker.tracker import CarbonTracker

class CarbonTrackerListener(threading.Thread):
    """
    CarbonTrackerListener handles the CarbonTracker instance and coordinates
    with the EpochListener to detect epoch transitions.
    """
    def __init__(self, run_id, epoch=0, experiment_id=88, carbon_log_dir='./carbon_logs'):
        super(CarbonTrackerListener, self).__init__()
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.epoch = epoch
        
        self.carbon_log_dir = carbon_log_dir
        self.max_epochs = int(os.getenv("RADT_MAX_EPOCH")) 

        os.makedirs(carbon_log_dir, exist_ok=True)
        
        # Initialize a stop event for clean thread termination
        self._stop_event = threading.Event()

        self.last_epoch_time = time()
        self.latest_log_file = None

        # Initialize the CarbonTracker instance
        self.carbon_tracker = CarbonTracker(
            epochs=self.max_epochs,
            log_dir=carbon_log_dir,
            update_interval=10
        )

    def _on_epoch_detected(self, new_epoch):
        """
        Callback called when a new epoch is detected.
        """
        # If you would like to detect a change (for example, new_epoch != previous epoch)
        # then additional state should be maintained if necessary.
        try:
            # If we're moving on from a previous epoch, stop it first.
            # In this example, we assume that if new_epoch > previous epoch,
            # then an epoch just ended.
            if new_epoch > 0:
                self.carbon_tracker.epoch_end()
                self.parse_and_log_metrics(new_epoch - 1)
        except Exception as e:
            print(f"Error ending epoch {new_epoch - 1}: {e}")

        try:
            self.carbon_tracker.epoch_start()
        except Exception as e:
            print(f"Error starting epoch {new_epoch}: {e}")

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
                mlflow.log_metrics(metrics, self.epoch.value)
                
        except Exception as e:
            print(f"Error parsing or logging metrics from file {self.latest_log_file}: {e}")

    def run(self):
        mlflow.start_run(run_id=self.run_id).__enter__()
        self.carbon_tracker.epoch_start()

        # Local copy of epoch to detect changes
        last_epoch = self.epoch.value

        try:
            while not self._stop_event.is_set() and self.epoch.value < self.max_epochs:
                # Check if the shared variable was updated
                if self.epoch.value != last_epoch:
                    self._on_epoch_detected(self.epoch.value)
                    last_epoch = self.epoch.value  # update local copy

                sleep(1)  # adjust sleep duration as necessary
        except Exception as e:
            print(f"Error in CarbonTrackerListener: {e}")
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

    def terminate(self):
        self._stop_event.set()
        try:
            self.carbon_tracker.epoch_end()
        except Exception:
            pass
        self.carbon_tracker.stop
