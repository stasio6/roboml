from h5py import File, Group, Dataset
import numpy as np

def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")

def load_hdf5(path):
    print('Loading HDF5 file', path)
    file = File(path, 'r')
    ret = load_content_from_h5_file(file)
    file.close()
    print('Loaded')
    return ret

TARGET_KEY_TO_SOURCE_KEY = {
    'states': 'env_states',
    'observations': 'obs',
    'success': 'success',
    # 'next_observations': 'next_obs',
    # 'dones': 'dones',
    'rewards': 'rewards',
    'actions': 'actions',
}

def load_demo_dataset(path, keys=['observations', 'actions'], num_traj=None):
    assert num_traj is None
    raw_data = load_hdf5(path)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    dataset = {}
    for target_key in keys:
        if 'next' in target_key:
            raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [ raw_data[idx][source_key] for idx in raw_data ]
        if isinstance(dataset[target_key][0], np.ndarray):
            if target_key in ['observations', 'states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                # dataset[target_key] = np.concatenate([
                #     t[:-1] for t in dataset[target_key]
                # ], axis=0)
                print("nope")
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            # print('Load', target_key, dataset[target_key].shape)
        else:
            print('Load', target_key, len(dataset[target_key]), type(dataset[target_key][0]))
    return dataset

def load_demo_dataset_with_state(path, keys=['observations', 'actions'], num_traj=None):
    assert num_traj is None
    raw_data = load_hdf5(path)
    dataset = {}
    for target_key in keys:
        if 'next' in target_key:
            raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [ raw_data[idx][source_key] for idx in raw_data ]
        if isinstance(dataset[target_key][0], np.ndarray):
            print('Load', target_key, len(dataset[target_key]), dataset[target_key][0].shape)
        else:
            print('Load', target_key, len(dataset[target_key]), type(dataset[target_key][0]))
    return dataset

def load_trajecories(path):
    raw_data = load_hdf5(path)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    return list(raw_data.values())

import json
def load_json(filename):
    filename = str(filename)
    if filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret

def load_trajecories_and_json(path):
    raw_data = load_hdf5(path)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    traj = list(raw_data.values())

    # Load associated json
    json_path = path.replace(".h5", ".json")
    json_data = load_json(json_path)

    return traj, json_data

if __name__ == "__main__":
    # path = 'data/pick_cube/trajectory.h5'
    path = 'data/pick_cube/trajectory.rgbd.pd_joint_delta_pos.h5'
    # load_trajecories(path)
    load_demo_dataset(path)