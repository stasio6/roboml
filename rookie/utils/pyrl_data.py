from h5py import File, Group, Dataset
import numpy as np
from sklearn import datasets

def load_hdf5_from_h5_file(file):
    if isinstance(file, (File, Group)):
        keys = list(file.keys())
        ret = {}
        for key in keys:
            if key.startswith("dict"):
                key_type = eval(key.split('_')[1])
                key_value = key_type(key[len(f"dict_{key.split('_')[1]}_"):]) # sth like "actions"
            else:
                key_value = key
            ret[key_value] = load_hdf5_from_h5_file(file[key])
        return ret
    elif isinstance(file, Dataset):
        ret = file[()]
        return ret

def load_hdf5(path):
    file = File(path, 'r')
    ret = load_hdf5_from_h5_file(file)
    file.close()
    return ret

TARGET_KEY_TO_SOURCE_KEY = {
    'actions': 'actions',
    'observations': 'obs',
    'next_observations': 'next_obs',
    'dones': 'dones',
    'rewards': 'rewards',
}

def load_demo_dataset(path, keys=['observations', 'actions']):
    raw_data = load_hdf5(path)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    dataset = {}
    for target_key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = np.concatenate([
            raw_data[idx][source_key] for idx in raw_data
        ], axis=0)
        print('Load', target_key, dataset[target_key].shape)
    return dataset

if __name__ == "__main__":
    path = '/home/tmu/pyrl/work_dirs/PickPanda_plus1_state-v0/SAC/20220225_123844_1M_buffer/eval_5000000/trajectory.h5'
    load_demo_dataset(path)