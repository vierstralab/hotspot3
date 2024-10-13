import psutil
import time
import subprocess

# Path to the output memory log file (tab-separated)
MEMORY_LOG = "memory_usage.tsv"

# Command to run your Python script
cmd = [
    "python3", "/home/sabramov/packages/hotspot2/hotspot2/main.py", "AG70782.test",
    "--bam", "../../filtered.cram",
    "--chrom_sizes", "/net/seq/data/genomes/human/GRCh38/noalts/GRCh38_no_alts.chrom_sizes",
    "--cutcounts", "AG70782.test.cutcounts.gz",
    "--mappable_bases", "/net/seq/data2/projects/sabramov/SuperIndex/GRCh38_no_alts.K36.center_sites.n100.nuclear.merged.bed.gz",
    "--cpus", "19",
    "--debug",
    "--fdrs", "0.1", "0.05",
    "--save_density"
]

def format_memory(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"

# Clear the log file before starting and write header with \t separation
with open(MEMORY_LOG, "w") as f:
    f.write("timestamp\ttotal_memory_rss_bytes\ttotal_memory_rss_human\n")

# Start the Python script as a subprocess and track memory usage
try:
    process = subprocess.Popen(cmd)
    python_process = psutil.Process(process.pid)

    while process.poll() is None:  # While the process is still running
        with open(MEMORY_LOG, "a") as log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            total_rss = python_process.memory_info().rss
            for child in python_process.children(recursive=True):
                if not child.is_running():
                    continue
                total_rss += child.memory_info().rss

            total_rss_human = format_memory(total_rss)
            log_file.write(f"{timestamp}\t{total_rss}\t{total_rss_human}\n")
        time.sleep(2)

except (psutil.NoSuchProcess, subprocess.SubprocessError, KeyboardInterrupt):
    print("Process interrupted or failed. Terminating...")

finally:
    if process.poll() is None:  # If the process is still running
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Forcefully killing the subprocess...")
            process.kill()

print("Memory tracking finished.")
