import h5py
import sys
import glob
import pickle

name = sys.argv[1]
outfile = "out"
if len(sys.argv) > 2:
    if sys.argv[2] == "--same-name":
        outfile = name.split(".")[0]
    else:
        print("Unrecognized third argument. Did you mean '--same-name'?")
        exit(0)

if name[-2:] == "h5":
    outfile += ".pkl"
    print("Output file:", outfile)
    with open(outfile,mode='wb') as output:
        traj = []
        h5fr = h5py.File(name,'r')
        print(h5fr)
        print(h5fr.keys())
        print(h5fr.get("traj_0"))
        print(h5fr["traj_0"])
        # exit(0)
        num = 0
        for traj_id in h5fr:
            arr_data = dict(h5fr[traj_id])
            num += 1
            print(num, traj_id, len(list(arr_data['actions'])))
            my_dict = {}
            for key in arr_data:
                target_key = key
                if target_key == 'obs':
                    target_key = 'observations'
                my_dict[target_key] = list(arr_data[key])
            traj.append(my_dict)
        print(num)
        pickle.dump(traj, output)

else:
    outfile += ".h5"
    print("Output file:", outfile)
    with open(name,mode='rb') as input:
        trajectories = pickle.load(input)
        h5fr = h5py.File(outfile,'w')
        num = 0
        for arr_data in trajectories:
            traj_id = "traj_" + str(num)
            num += 1
            print(num, traj_id, len(list(arr_data['actions'])))
            my_dict = {}
            for key in arr_data:
                target_key = key
                if target_key == 'observations':
                    target_key = 'obs'
                my_dict[target_key] = list(arr_data[key])
            gr = h5fr.create_group(traj_id)
            for key in my_dict:
                if key == "infos":
                    continue
                gr.create_dataset(key, data=my_dict[key])
        print(num)