import os
rootdir = '/cephfs_fast/output'

for subdir, dirs, files in os.walk(rootdir):
    ile = 0
    for file in files:
        if file[-4:] == ".mp4":
            print(os.path.join(subdir, file))
            os.remove(os.path.join(subdir, file))
print(ile)