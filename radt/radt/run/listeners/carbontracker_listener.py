import os
import re
import glob
import mlflow
import threading
from time import sleep, time 
from carbontracker.tracker import CarbonTracker

class CarbonTrackerListener(threading.Thread):
    
    def __init__(self, run_id, experiment_id=88, carbon_log_dir='./carbon_logs', epoch_duration_threshold=2):
        super().__init__(daemon=True)
        self.run_id = run_id
        self.carbon_log_dir = carbon_log_dir
        self.epoch_duration_threshold = epoch_duration_threshold
        self.max_epochs = int(os.getenv("RADT_MAX_EPOCH"))

        # Internal state tracking
        self.current_epoch = 0
        self.last_epoch_time = time()
        self._stop_event = threading.Event()
        self.latest_log_file = None  # Store the latest log file path

        # Initialize the CarbonTracker instance
        self.carbon_tracker = CarbonTracker(epochs=int(os.getenv("RADT_MAX_EPOCH")), log_dir=carbon_log_dir)

    def parse_and_log_metrics(self, epoch):
        log_files = glob.glob(os.path.join(self.carbon_log_dir, '*.log'))
        if not log_files:
            print("No CarbonTracker log files found yet.")
            return

        # Use the most recently created log file
        self.latest_log_file = max(log_files, key=os.path.getctime)

        try:
            with open(self.latest_log_file, 'r') as f:
                log_content = f.read()  # Read entire file content

            # Find all GPU power usage entries
            power_matches = re.findall(r'Epoch\s*(\d+):.*?Average power usage \(W\) for gpu:\s*(\d+\.?\d*)', log_content, re.DOTALL | re.IGNORECASE)
            
            # Find carbon intensity value
            carbon_intensity_match = re.search(r'Average carbon intensity during training was (\d+\.?\d*) gCO2/kWh', log_content, re.IGNORECASE)
            
            # Prepare metrics dictionary
            metrics = {}

            # Log GPU power for the current epoch
            for matched_epoch, gpu_power in power_matches:
                matched_epoch = int(matched_epoch)
                
                # Only log the metric for the current processing epoch
                if matched_epoch == epoch:
                    metrics['Carbontracker - GPU Power W'] = float(gpu_power)

            # Log overall carbon intensity if found
            if carbon_intensity_match:
                carbon_intensity = float(carbon_intensity_match.group(1))
                metrics['Carbontracker - Carbon Intensity gCO2/kWh'] = carbon_intensity

            # Log metrics for the current processing epoch
            if metrics:
                mlflow.log_metrics(metrics, step=epoch)
                
        except Exception as e:
            print(f"Error parsing or logging metrics from file {self.latest_log_file}: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        mlflow.start_run(run_id=self.run_id).__enter__()  # Attach to run
        self.carbon_tracker.epoch_start()  # Start CarbonTracker monitoring

        try:
            while not self._stop_event.is_set() and self.current_epoch < self.max_epochs:
                time_now = time()
                time_since_last_epoch = time_now - self.last_epoch_time

                # Check if enough time has passed for a new epoch
                if time_since_last_epoch >= self.epoch_duration_threshold:
                    self.current_epoch += 1
                    self.last_epoch_time = time_now

                    # End the previous epoch, start the next one, and log metrics
                    try:
                        self.carbon_tracker.epoch_end()
                        self.carbon_tracker.epoch_start()
                        self.parse_and_log_metrics(self.current_epoch)
                    except Exception as e:
                        print(f"Error during epoch {self.current_epoch}: {e}")

                # Sleep to prevent tight looping
                sleep(5)

        except Exception as e:
            print(f"Error in CarbonTracker listener: {e}")
        finally:
            # Ensure CarbonTracker and MLflow are stopped cleanly
            self.carbon_tracker.epoch_end()
            
            # Log the final complete log file as an artifact
            if self.latest_log_file and os.path.exists(self.latest_log_file):
                mlflow.log_artifact(self.latest_log_file, artifact_path='carbon_logs')
            
            self.carbon_tracker.stop()
            mlflow.end_run()

    def terminate(self):
        self._stop_event.set()
        self.carbon_tracker.epoch_end()
        self.carbon_tracker.stop()
