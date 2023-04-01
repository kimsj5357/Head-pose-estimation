## Amir Shahroudy  
# https://github.com/shahroudy

import os, sys, argparse
import numpy as np
from utils import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Create filenames list txt file from datasets root dir.'
        ' For head pose analysis.')
    parser.add_argument('--root_dir = ', 
        dest='root_dir', 
        help='root directory of the datasets files', 
        default='./datasets/300W_LP',
        type=str)
    parser.add_argument('--filename', 
        dest='filename', 
        help='Output filename.',
        default='files.txt', 
        type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    os.chdir(args.root_dir)

    file_counter = 0
    rej_counter = 0

    filelist_txt = ''
    for root, dirs, files in os.walk('.'):
        if 'Flip' in root:
            continue
        for f in files: 
            if f[-4:] == '.jpg': 
                mat_path = os.path.join(root, f.replace('.jpg', '.mat'))
                # We get the pose in radians
                pose = utils.get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi

                if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                    if file_counter > 0:
                        filelist_txt += '\n'
                    filelist_txt += root + '/' + f[:-4]
                    file_counter += 1
                else:
                   rej_counter += 1


    outfile = open(args.filename, 'w')

    outfile.write(filelist_txt)

    outfile.close()
    print(f'{file_counter} files listed! {rej_counter} files had out-of-range'
        f' values and kept out of the list!')