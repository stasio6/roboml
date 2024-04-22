import os
rootdir = '/cephfs_fast/output'

ile = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == ".mp4":
            print(os.path.join(subdir, file))
            os.remove(os.path.join(subdir, file))
            ile += 1
print(ile)