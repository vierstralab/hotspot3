import psutil
import time
import subprocess
import sys
from main import parse_arguments
import os
"""
This script is used to call main.py with memory tracking.
"""

def format_memory(size_in_bytes):
    """Format the memory size from bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"



def track_memory(process, log_file, interval=2):
    """Track memory usage of a subprocess and log it."""
    python_process = psutil.Process(process.pid)
    with open(log_file, "w") as f:
        f.write("timestamp\ttotal_memory_rss_bytes\ttotal_memory_rss_human\n")
    
    while process.poll() is None:  # While the process is still running
        with open(log_file, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            total_rss = python_process.memory_info().rss
            for child in python_process.children(recursive=True):
                if not child.is_running():
                    continue
                total_rss += child.memory_info().rss

            total_rss_human = format_memory(total_rss)
            f.write(f"{timestamp}\t{total_rss}\t{total_rss_human}\n")
        time.sleep(interval)


def run_process_with_memory_tracking(cmd, log_file):
    """Run a subprocess and track its memory usage."""
    try:
        process = subprocess.Popen(cmd)  # Start the process
        track_memory(process, log_file)  # Start tracking memory usage
    except (psutil.NoSuchProcess, subprocess.SubprocessError, KeyboardInterrupt):
        print("Process interrupted or failed. Terminating...")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Forcefully killing the subprocess...")
                process.kill()


def main():
    args, _ = parse_arguments(" with memory tracking. Outputs {args.id}.memory_usage.tsv in output folder.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["python3", f"{script_dir}/main.py", *sys.argv[1:]]

    memory_log = f"{args.outdir}/{args.id}.memory_usage.tsv"
    run_process_with_memory_tracking(cmd, memory_log)

    print("Memory tracking finished.")


if __name__ == "__main__":
    main()
