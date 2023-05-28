import copy
import pickle 
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Update data root path of nuscenes infos files produced by prepare_data.py'
    )
    parser.add_argument('info_file', help='info file path')
    parser.add_argument('data_root', help='root path of data folder')
    return parser.parse_args()


def update_path(path, new_data_root):
    if new_data_root[-1] != '/':
        new_data_root += '/'
    return new_data_root + path.split('/data/')[1]


def main(info_file, data_root):
    print('Updating data root path ...')
    with open(info_file, 'rb') as f:
        infos = pickle.load(f)
        updated_infos = copy.deepcopy(infos)
        
        for idx, info in enumerate(infos['infos']):
            updated_infos['infos'][idx]['lidar_path'] = update_path(info['lidar_path'], data_root)
            for key in info['cams'].keys():
                updated_infos['infos'][idx]['cams'][key]['data_path'] = update_path(info['cams'][key]['data_path'], data_root)
    
    with open(info_file, 'wb') as f:
        pickle.dump(updated_infos, f)


if __name__ == '__main__':
    args = parse_args()
    
    main(args.info_file, args.data_root)
