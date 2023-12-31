import sys
sys.path.append('..')
from libs.global_variable import ROOT_DIR

import os
import argparse
import pickle as pkl
import random

def train_val_data_split(train_subject, data_path, save_loc):

    data = {'train': [], 'val': []}

    train_frames = []
    val_frames = []

    for subject_idx in train_subject:

        sequences_path = data_path + 'sequences/' + str(subject_idx)
        
        subject_available_frames = []

        for action in os.listdir(sequences_path):
            action_path = os.path.join(sequences_path, action)

            for npy_file_name in os.listdir(action_path):
            
                if npy_file_name.split('.')[0].endswith('cano'):
                    continue
                frame_npy_path = os.path.join(action_path, npy_file_name)

                if not os.path.exists(frame_npy_path):
                    print('files missing of', subject_idx, " ", npy_file_name)
                    continue
                frame_relative_path = os.path.relpath(frame_npy_path, ROOT_DIR)
                subject_available_frames.append(frame_relative_path)

            print(len(subject_available_frames))
            random.shuffle(subject_available_frames)
            subject_available_frames = subject_available_frames[::4]
            count = int(len(subject_available_frames)*0.9)

            train_frames.extend(subject_available_frames[:count])
            val_frames.extend(subject_available_frames[count:])

    data['train'] = train_frames
    data['val'] = val_frames

    print('train={}, val={}'.format(len(data['train']), len(data['val'])))

    pkl.dump(data, open(save_loc, 'wb')) # write every 10th training data and all evaluation data to the saving directory


def animation_data_split(train_subject, data_path, save_loc):

    data = {'train': [], 'val': []}

    subject_available_frames = []

    for subject_idx in train_subject:

        sequences_path = data_path + 'sequences/' + str(subject_idx)
        print(sequences_path)

        for action in os.listdir(sequences_path):
            action_path = os.path.join(sequences_path, action)

            for npy_file_name in os.listdir(action_path):
                
                if npy_file_name.split('.')[0].endswith('cano'):
                    continue
                frame_npy_path = os.path.join(action_path, npy_file_name)

                if not os.path.exists(frame_npy_path):
                    print('files missing of', subject_idx, " ", npy_file_name)
                    continue
                frame_relative_path = os.path.relpath(frame_npy_path, ROOT_DIR)
                subject_available_frames.append(frame_relative_path)

            print(len(subject_available_frames))

    data['val'] = subject_available_frames        # for fine tune

    print('train={}, val={}'.format(len(data['train']), len(data['val'])))

    pkl.dump(data, open(save_loc, 'wb')) # write every 10th training data and all evaluation data to the saving directory




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Model')
    # experiment id for folder suffix
    parser.add_argument('-gender', '--gender', type=str)
    parser.add_argument('-mode', '--mode', type=str)

    args = parser.parse_args()

    preprocessed_path = ROOT_DIR + 'Data/BuFF/buff_release_rot_const/'

    if args.gender == 'male':
        training_subject = ["00096", "00032"]

        if args.mode == 'train':
            save_loc = ROOT_DIR + 'assets/data_split/buff_male_train_val.pkl'
        elif args.mode == 'animation':
            save_loc = ROOT_DIR + 'assets/data_split/buff_male_animation.pkl'

    elif args.gender == 'female':
        training_subject = ["03223"]

        if args.mode == 'train':
            save_loc = ROOT_DIR + 'assets/data_split/buff_female_train_val.pkl'
        elif args.mode == 'animation':
            save_loc = ROOT_DIR + 'assets/data_split/buff_female_animation.pkl'

    if not os.path.exists(os.path.dirname(save_loc)):
        os.makedirs(os.path.dirname(save_loc))

    if args.mode == 'train':
        train_val_data_split(training_subject, preprocessed_path, save_loc)
    elif args.mode == 'animation':
        animation_data_split(training_subject, preprocessed_path, save_loc)
