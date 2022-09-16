import os
root = '/root/softgym/data/rope-configuration-tryone-step1-U'
# root = './data'
dirs = os.listdir(root)
get_key = lambda i : int(i.split('-')[0])
new_sort = sorted(dirs, key=get_key)

for fname in new_sort:
    path = os.path.join(root, fname)
    print(fname)
    name_id = int(fname.split('-')[0]) - 3000
    new_fname = f'{name_id:06d}' + '-1.pkl'
    print(new_fname)
    new_path = os.path.join(root, new_fname)
    os.rename(path, new_path)
