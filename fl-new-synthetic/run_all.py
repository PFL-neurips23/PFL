import os
import glob
import shutil
import re
# Change this to your base directory containing the data folders
base_dir = './data_files/Synthetic'

# Get a list of all the data folders
data_folders = glob.glob(os.path.join(base_dir, 'data_alpha_*'))

# Define the base command to run the experiments
command_base = "python3 main.py {args} -out transfer/ten_worker_equal_final/{subdir}/{placeholder}.csv"

# Regular expression to extract the parameter values
pattern = r"data_alpha_(\d+\.\d+)_beta_(\d+\.\d+)_iid_(\d+)_lnf_(\d+\.\d+)_rdf_(\d+\.\d+)_lns_(\d+\.\d+)_rdfs_(\d+\.\d+)"

# Loop through the folders
# Define the selection methods
selection_methods = ["None", "All", "Ada"]

# Loop through the folders
for i, folder in enumerate(data_folders):
    # Extract the parameters from the folder name
    match = re.search(pattern, folder)
    alpha, beta, iid, lnf, rdf, lns, rdfs = map(float, match.groups())
    iid = int(iid)
    # Create a subdirectory with a unique name for the experimeßnt
    subdir = f"experiment_alpha_{alpha}_beta_{beta}_iid_{iid}_lnf_{lnf}_rdf_{rdf}_lns_{lns}_rdfs_{rdfs}"
    if os.path.exists(f"transfer/ten_worker_equal_final/{subdir}"):
        shutil.rmtree(f"transfer/ten_worker_equal_final/{subdir}")  # Delete the contents of the directory
    os.makedirs(f"transfer/ten_worker_equal_final/{subdir}")
    for method in selection_methods:
        # Create the experiment string with the new arguments
        experiment = f"-data synthetic -availability always -seeds 1,2,3,4,5 -lr-warmup 0.1 -iters-warmup 100 -iters-total 1500 -lr 0.01 -lr-global 1.0 -wait-all 0 -full-batch 0 -total-workers_w 10 -similarity_w 0 --alpha {alpha} --beta {beta} --iid {iid} --label_noise_factor {lnf} --random_data_fraction_factor {rdf} --label_noise_skew_factor {lns} --random_data_fraction_skew_factor {rdfs} -selection-method {method}"



        # Build and run the command with the subdirectory and unique identifier
        command = command_base.format(args=experiment, subdir=subdir, placeholder="results")
        os.system(command)
    break
