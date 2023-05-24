import os
import shutil

# Define a list of experiments to run
experiments = [
#     "-data synthetic -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 0",     #Baseline
#     "-data synthetic -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 500 -lr 0.001 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0", #workers are always 10
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.1",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.2",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.3",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.4",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.5",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.6",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.7",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.8",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0.9",
#     "-data fashion -availability always -seeds 1 -lr-warmup 0.1 -iters-warmup 0 -iters-total 1500 -lr 0.1 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 1.0",
    # Add more experiments here as needed
]

# Define the base command to run the experiments
# os.makedirs(f"transfer/test")
command_base = "python3 main.py {args} -out transfer/ten_worker_equal_split/{subdir}/{placeholder}.csv"

# # Loop through each experiment and run the command
# for i, experiment in enumerate(experiments):
#     command = command_base.format(args=experiment, placeholder=f"experiment_{i+1}/experiment_{i+1}")
#     os.system(command)
for i, experiment in enumerate(experiments):
    # Create a subdirectory with a unique name for the experiment
    subdir = f"experiment_{i+1}"
    if os.path.exists(f"transfer/ten_worker_synth/{subdir}"):
        shutil.rmtree(f"transfer/ten_worker_synth/{subdir}")  # Delete the contents of the directory
    os.makedirs(f"transfer/ten_worker_equal_synth/{subdir}")

    # Build and run the command with the subdirectory and unique identifier
    command = command_base.format(args=experiment, subdir=subdir, placeholder="results")
    os.system(command)