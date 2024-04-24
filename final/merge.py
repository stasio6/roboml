import h5py
import sys
import glob
import pickle

# name = "pick_20_gail_merged"
# name_pkl = "checkpoints/PegInsertionSide-v1/evaluation/PegInsertionSide-v1/PegInsertionSide-v1_trajectories_20.pkl"

# with open("ckpts/"+name+'.pkl',mode='wb') as output:
#     traj = []
#     fname = "ckpts/"+name+".h5"
#     is35 = False
#     h5fr = h5py.File(fname,'r')
#     num = 0
#     completed = 0
#     for traj_id in h5fr:
#         arr_data = dict(h5fr[traj_id])
#         # print(list(arr_data['actions']))
#         num += 1
#         if len(list(arr_data['actions'])) > 35 and is35:
#             continue
#         if len(list(arr_data['actions'])) >= 95:
#             continue
#         completed += 1
#         print(num, traj_id, len(list(arr_data['actions'])))
#         my_dict = {}
#         for key in arr_data:
#             target_key = key
#             if target_key == 'obs':
#                 target_key = 'observations'
#             my_dict[target_key] = list(arr_data[key])
#         print(my_dict.keys())
#         traj.append(my_dict)
#     # traj = traj[:-1]
#     print(num, completed)
#     with open(name_pkl, "rb") as pkl_in:
#         data = pickle.load(pkl_in)
#         traj.extend(data)
#     print(len(traj))
#     pickle.dump(traj, output)

name = "faucet_sparse_merged"
with open("ckpts/"+name+'.pkl',mode='wb') as output:
    traj = []
    fname = "checkpoints/TurnFaucet-v0/evaluation/TurnFaucet-v0/TurnFaucet-v0_trajectories_200.pkl"
    name_pkl = "ckpts/faucet_sparse.pkl"
    with open(fname, "rb") as fpkl_in:
        data = pickle.load(fpkl_in)
        traj.extend(data)
    with open(name_pkl, "rb") as pkl_in:
        data = pickle.load(pkl_in)
        traj.extend(data)
    print(len(traj))
    pickle.dump(traj, output)


# with open("ckpts/"+name+'ul.pkl',mode='wb') as output:
#     traj = []
#     # fname = "checkpoints_gail/PickCube-v1/evaluation/PickCube-v1_50/PickCube-v1_trajectories_300.pkl"
#     # with open(fname, "rb") as fpkl_in:
#         # data = pickle.load(fpkl_in)
#         # traj.extend(data)
#     with open("ckpts/"+name+'.pkl',mode='rb') as input:
#         print(input)
#         data = pickle.load(input)
#         new_data = []
#         for traj in data:
#             new_traj = {}
#             for key in traj:
#                 target_key = key
#                 if target_key == 'obs':
#                     target_key = 'observations'
#                 new_traj[target_key] = traj[key]
#             new_data.append(new_traj)
#         # for t in traj:
#             # if len(t) != 51:
#                 # print("nono")
#         # traj.extend(data)
#     # print(len(traj))
#     pickle.dump(new_traj, output)
    