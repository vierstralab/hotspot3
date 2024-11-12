import psutil
import time
import subprocess
import sys
import os
import threading
import logging

from hotspot3.main import parse_arguments
from hotspot3.io.logging import setup_logger

"""
This script is used to call main.py with memory tracking.
Inefficient to use this script in a pipeline, didn't optimize it too much.
Also, time estimates from logger might be inaccurate due to output buffering.
"""


def format_memory(size_in_bytes):
    """Format the memory size from bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"


def track_memory(process, logger: logging.Logger, interval=2):
    """Track memory usage of a subprocess and log it."""
    python_process = psutil.Process(process.pid)
    logger.info("total_memory_rss_bytes\ttotal_memory_rss_human")
    
    while process.poll() is None:  # While the process is still running
    
        total_rss = python_process.memory_info().rss
        for child in python_process.children(recursive=True):
            if not child.is_running():
                continue
            total_rss += child.memory_info().rss

        total_rss_human = format_memory(total_rss)
        logger.info(f"{total_rss}\t{total_rss_human}")
        time.sleep(interval)


def run_process_with_memory_tracking(cmd, logger: logging.Logger):
    """Run a subprocess and track its memory usage."""
    try:
        process = subprocess.Popen(cmd)  # Start the process

        memory_thread = threading.Thread(target=track_memory, args=(process, logger))
        memory_thread.start()
        process.wait()
        memory_thread.join()
    except (psutil.NoSuchProcess, subprocess.SubprocessError, KeyboardInterrupt):
        logger.critical("Process interrupted or failed. Terminating...")
        raise
    finally:
        if memory_thread.is_alive():
            memory_thread.join()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                raise
        

def main():
    args, logger_level = parse_arguments(" with memory tracking. Creates {args.id}.memory_usage.tsv in output folder.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = ["python3", f"{script_dir}/main.py", *sys.argv[1:]]

    memory_log = f"{args.outdir}/{args.id}.memory_usage.tsv"
    logger = setup_logger(level=logger_level, outstream=memory_log)
    run_process_with_memory_tracking(cmd, logger)


if __name__ == "__main__":
    main()
