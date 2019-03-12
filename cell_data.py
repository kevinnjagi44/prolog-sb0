import os
import glob
import shutil

path = '/root/datasets/CellOCT2017'

cvn_list = sorted(glob.glob(os.path.join(path, 'train/CNV/*.jpeg')))
dme_list = sorted(glob.glob(os.path.join(path, 'train/DME/*.jpeg')))
drusen_list = sorted(glob.glob(os.path.join(path, 'train/DRUSEN/*.jpeg')))
normal_list = sorted(glob.glob(os.path.join(path, 'train/NORMAL/*.jpeg')))
# for i in range(500):
#     print(i)
#     shutil.move(cvn_list[i], os.path.join(path, 'val/CNV', os.path.split(cvn_list[i])[1]))
#     shutil.move(dme_list[i], os.path.join(path, 'val/DME', os.path.split(dme_list[i])[1]))
#     shutil.move(drusen_list[i], os.path.join(path, 'val/DRUSEN', os.path.split(drusen_list[i])[1]))
#     shutil.move(normal_list[i], os.path.join(path, 'val/NORMAL', os.path.split(normal_list[i])[1]))


for i in range(250):
    print(i)
    shutil.move(cvn_list[i], os.path.join(path, 'test/CNV', os.path.split(cvn_list[i])[1]))
    shutil.move(dme_list[i], os.path.join(path, 'test/DME', os.path.split(dme_list[i])[1]))
    shutil.move(drusen_list[i], os.path.join(path, 'test/DRUSEN', os.path.split(drusen_list[i])[1]))
    shutil.move(normal_list[i], os.path.join(path, 'test/NORMAL', os.path.split(normal_list[i])[1]))
