import os
import threading
import time
import mlflow
from carbontracker.tracker import CarbonTracker

class CarbonTrackerThread(threading.Thread):
    """Thread for tracking carbon emissions using carbontracker."""
    
    def __init__(self, run_id, interval=60):
        """Initialize the CarbonTracker thread.
        
        Args:
            run_id (str): MLflow run ID (stored but not used for logging)
            interval (int): Interval in seconds between tracking measurements
        """
        threading.Thread.__init__(self)
        self.run_id = run_id
        self.interval = interval
        self.terminated = False
        
        # Get max epochs from environment or default to 5
        self.max_epochs = int(os.getenv("RADT_MAX_EPOCH", 5))
        
        # Set up log directory
        self.log_dir = os.getenv("RADT_CARBON_LOG_DIR", "./carbon_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize the tracker with file logging enabled
        self.tracker = CarbonTracker(
            epochs=self.max_epochs,
            log_dir=self.log_dir,
            update_interval=self.interval,
            verbose=1          # Show more detailed output
        )
        
        # Current epoch tracking
        self.current_epoch = 0
        
        # For tracking whether we're in an epoch
        self.in_epoch = False
        
        # Store log file paths
        self.standard_log = None
        self.output_log = None
        
    def run(self):
        """Run the thread."""
        try:
            # Start the global tracker
            self.tracker.init()
            
            # Start the first epoch
            self.epoch_start()
            
            while not self.terminated:
                # Just sleep for the interval - logging is handled by carbontracker
                time.sleep(self.interval)
        except Exception as e:
            print(f"Error in CarbonTracker thread: {e}")
        finally:
            # Clean up
            if self.tracker:
                self.tracker.stop()
                
                # After stopping, log the results summary to MLflow as an artifact
                try:
                    # Find the log files that were created
                    from carbontracker import parser
                    logs = parser.parse_all_logs(log_dir=self.log_dir)
                    
                    if logs:
                        # Create a summary file with the results
                        summary_path = os.path.join(self.log_dir, f"carbon_summary_{self.run_id}.txt")
                        with open(summary_path, "w") as f:
                            for log in logs:
                                f.write(f"Log file: {log['output_filename']}\n")
                                f.write(f"Duration: {log['actual']['duration (s)']} seconds\n")
                                f.write(f"Energy used: {log['actual']['energy (kWh)']} kWh\n")
                                f.write(f"CO2 emissions: {log['actual']['co2eq (g)']} g\n")
                                f.write(f"Equivalent to {log['actual']['equivalents']['km travelled by car']} km travelled by car\n")
                                f.write("\n")
                            
                            # Add the aggregated totals
                            f.write("--- AGGREGATED TOTALS ---\n")
                            total_energy = sum(log['actual']['energy (kWh)'] for log in logs)
                            total_co2 = sum(log['actual']['co2eq (g)'] for log in logs)
                            total_km = sum(log['actual']['equivalents']['km travelled by car'] for log in logs)
                            
                            f.write(f"Total energy: {total_energy} kWh\n")
                            f.write(f"Total CO2: {total_co2} g ({total_co2/1000} kg)\n")
                            f.write(f"Equivalent to {total_km} km travelled by car\n")
                        
                        # Log the summary file to MLflow
                        mlflow.log_artifact(summary_path)
                except Exception as e:
                    print(f"Error creating carbon summary: {e}")
    
    def epoch_start(self):
        """Mark the start of an epoch."""
        if not self.in_epoch:
            self.in_epoch = True
            self.tracker.epoch_start()
    
    def epoch_end(self):
        """Mark the end of an epoch."""
        if self.in_epoch:
            self.tracker.epoch_end()
            self.in_epoch = False
            self.current_epoch += 1
            
            # Start the next epoch if we haven't reached the maximum
            if self.current_epoch < self.max_epochs:
                self.epoch_start()
    
    def terminate(self):
        """Terminate the thread."""
        self.terminated = True
        if self.tracker:
            self.tracker.stop()
            
    def get_summary(self):
        """Get a summary of the carbon tracking results.
        
        Returns:
            dict: Summary of carbon tracking results
        """
        try:
            from carbontracker import parser
            logs = parser.parse_all_logs(log_dir=self.log_dir)
            return logs
        except Exception as e:
            print(f"Error getting carbon summary: {e}")
            return None
            
    def print_aggregate(self):
        """Print the aggregate carbon tracking results."""
        try:
            from carbontracker import parser
            parser.print_aggregate(log_dir=self.log_dir)
        except Exception as e:
            print(f"Error printing carbon aggregate: {e}")