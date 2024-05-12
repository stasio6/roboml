import h5py
import sys
import glob
import pickle

env = sys.argv[1]
num_traj = sys.argv[2]
name1 = "checkpoints/" + env + "/evaluation/" + env + "/" + env + "_trajectories_" + num_traj + "_with_reward.h5"
name2 = "traj_icem/" + env + "/icem_from_" + num_traj + ".h5"
outfile = "traj_icem/" + env + "/merged_" + num_traj + ".h5"

print("Output file:", outfile)
output = h5py.File(outfile,'w')
file1 = h5py.File(name1,'r')
file2 = h5py.File(name2,'r')
num = 0
for file in [file1, file2]:
    print(file)
    for traj_id in file:
        # print(traj_id)
        arr_data = dict(file[traj_id])
        # print(arr_data)
        n_traj_id = "traj_" + str(num)
        gr = output.create_group(n_traj_id)
        for key in arr_data:
            gr.create_dataset(key, data=arr_data[key])
        num += 1
print("Num total:", num)

# Not merged due to corruption
# Peg 20
# Pick 200
# Faucet 200