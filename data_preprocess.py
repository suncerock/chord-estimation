import csv
import pickle
import os
import math
from tqdm import tqdm

import numpy as np
from scipy.ndimage import gaussian_filter1d

import time


def enharmonic(chord):
    enharmonic_table = {'Cb': 'B',
                        'Db': 'C#',
                        'Eb': 'D#',
                        'Fb': 'E',
                        'Gb': 'F#',
                        'Ab': 'G#',
                        'Bb': 'A#'}

    root = chord.split(':')[0]
    quality = chord.split(':')[1]
    new_chord = enharmonic_table[root] + ':' + quality

    return new_chord


def compute_Tonal_centroids(chromagram, filtering=True, sigma=8):
    """chromagram with shape [time, 12] """

    # define transformation matrix - phi
    Pi = math.pi
    r1, r2, r3 = 1, 1, 0.5
    phi_0 = r1 * np.sin(np.array(range(12)) * 7 * Pi / 6)
    phi_1 = r1 * np.cos(np.array(range(12)) * 7 * Pi / 6)
    phi_2 = r2 * np.sin(np.array(range(12)) * 3 * Pi / 2)
    phi_3 = r2 * np.cos(np.array(range(12)) * 3 * Pi / 2)
    phi_4 = r3 * np.sin(np.array(range(12)) * 2 * Pi / 3)
    phi_5 = r3 * np.cos(np.array(range(12)) * 2 * Pi / 3)
    phi_ = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]
    phi = np.concatenate(phi_).reshape(6, 12)  # [6, 12]
    phi_T = np.transpose(phi)  # [12, 6]

    TC = chromagram.dot(phi_T)  # convert to tonal centiod representations, [time, 6]
    if filtering:  # Gaussian filtering along time axis
        TC = gaussian_filter1d(TC, sigma=sigma, axis=0)

    return TC.astype(np.float32)


def chord2int(chord):
    table = {'C:maj': 0, 'C#:maj': 1, 'D:maj': 2, 'D#:maj': 3, 'E:maj': 4, 'F:maj': 5,
             'F#:maj': 6, 'G:maj': 7, 'G#:maj': 8, 'A:maj': 9, 'A#:maj': 10, 'B:maj': 11,
             'C:min': 12, 'C#:min': 13, 'D:min': 14, 'D#:min': 15, 'E:min': 16, 'F:min': 17,
             'F#:min': 18, 'G:min': 19, 'G#:min': 20, 'A:min': 21, 'A#:min': 22, 'B:min': 23,
             'N': 24, 'X': 25}

    return table[chord]


def load_billboard(file_path):
    print('Running Message: read Billborad data ...')

    annotation_path = os.path.join(file_path, 'simple_lab')
    annotations = {}
    adt = [('onset', np.float32), ('end', np.float32), ('chord', object)]

    print("Loading annotations ...")
    for song_id in tqdm(os.listdir(annotation_path)):
        annotation = []
        path = os.path.join(annotation_path, song_id, 'majmin.lab')
        with open(path, 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                a = line.strip().split()
                assert len(a) == 3, (song_id, line)
                annotation.append((float(a[0]), float(a[1]), a[2]))
        annotations[song_id] = np.array(annotation, dtype=adt)

    print("Loading features ...")
    feature_path = os.path.join(file_path, 'chordino')
    BillboardData = {}
    dt = [('op', object), ('onset', np.float32), ('chroma', object), ('chord', np.int32), ('chordChange', np.int32)]
    for song_id in tqdm(os.listdir(feature_path)):
        path = os.path.join(feature_path, song_id, 'bothchroma.csv')
        rows = np.genfromtxt(path, delimiter=',')
        frames = []
        pre_chord = None
        for r, row in enumerate(rows):
            onset = row[1]
            chroma1 = row[2:14]
            chroma2 = row[14:26]
            both_chroma = np.concatenate([chroma1, chroma2]).astype(np.float32)

            label = annotations[song_id][
                (annotations[song_id]['onset'] <= onset) & (annotations[song_id]['end'] > onset)]
            if len(label) == 0:
                continue
            chord = label['chord'][0]
            root = chord.split(':')[0]
            if 'b' in root:
                chord = enharmonic(chord)
            chord_int = chord2int(chord)
            chordChange = 0 if chord_int == pre_chord else 1
            pre_chord = chord_int

            frames.append((song_id, onset, both_chroma, chord_int, chordChange))

        BillboardData[song_id] = np.array(frames, dtype=dt)

    with open('preprocessed_data/Billboard_data_mirex_Mm.pkl', 'wb') as output:
        pickle.dump(BillboardData, output)


def augment_Billboard():
    if os.path.exists('preprocessed_data/Billboard_data_mirex_Mm.pkl'):
        with open('preprocessed_data/Billboard_data_mirex_Mm.pkl', 'rb') as input_data:
            BillboardData = pickle.load(input_data)
    else:
        BillboardData = load_billboard('./billboard')

    def shift_chromagram(chromagram, shift):
        if shift > 0:
            chr1 = np.roll(chromagram[:, :12], shift, axis=1)
            chr2 = np.roll(chromagram[:, 12:], shift, axis=1)
            chromagram = np.concatenate([chr1, chr2], axis=1)
        return chromagram

    def shift_chord(chord, shift):
        if chord < 12:
            new_chord = (chord + shift) % 12
        elif chord < 24:
            new_chord = (chord - 12 + shift) % 12 + 12
        else:
            new_chord = chord
        return new_chord

    for shift in range(12):
        BillboardData_shift = {}
        for key, value in BillboardData.items():
            chromagram = np.array([x for x in value['chroma']])
            chord = value['chord']
            chordChange = value['chordChange']

            chromagram_shift = shift_chromagram(chromagram, shift)
            TC_shift = compute_Tonal_centroids((chromagram_shift[:, :12] + chromagram_shift[:, 12:])/2) # [time, 6]
            chord_shift = np.array([shift_chord(x, shift) for x in chord])
            chordChange_shift = chordChange

            BillboardData_shift[key] = {}
            BillboardData_shift[key]['chroma'] = chromagram_shift
            BillboardData_shift[key]['TC'] = TC_shift
            BillboardData_shift[key]['chord'] = chord_shift
            BillboardData_shift[key]['chordChange'] = chordChange_shift

        outputdir = 'preprocessed_data/Billboard_data_mirex_Mm_shift_{}.pkl'.format(shift)
        with open(outputdir, "wb") as output_file:
            pickle.dump(BillboardData_shift, output_file)


def segment_Billboard(segment_width=21, segment_hop=5):
    print('Running Message: segment Billborad ...')

    for shift in range(12):
        inputdir = 'preprocessed_data/Billboard_data_mirex_Mm_shift_' + str(shift) + '.pkl'
        with open(inputdir, 'rb') as input_data:
            BillboardData_shift = pickle.load(input_data)
        """BillboardData_shift = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array}, ...}"""

        BillboardData_shift_segment = {}
        for key, value in BillboardData_shift.items():
            chroma = value['chroma'] # [time, 24]
            TC = value['TC'] # [time, 6]
            chroma_TC = np.concatenate([chroma, TC], axis=1) # [time, 30]
            del chroma, TC
            chord = value['chord'] # [time,]

            n_pad = segment_width//2
            chroma_TC_pad = np.pad(chroma_TC, ((n_pad, n_pad), (0, 0)), 'constant', constant_values=0.0) # [time + 2*n_pad, 30]
            chord_pad = np.pad(chord, (n_pad, n_pad), 'constant', constant_values=24) # [time + 2*n_pad,]

            n_frames = chroma_TC.shape[0]
            chroma_TC_segment = np.array([chroma_TC_pad[i-n_pad:i+n_pad+1] for i in range(n_pad, n_pad + n_frames, segment_hop)]) # [n_segments, segment_width, 30]
            chroma_segment = np.reshape(chroma_TC_segment[:, :, :24], [-1, segment_width*24]) # [n_segments, segment_widt*24]
            TC_segment = np.reshape(chroma_TC_segment[:, :, 24:], [-1, segment_width*6]) # [n_segments, segment_widt*6]
            chord_segment = np.array([chord_pad[i] for i in range(n_pad, n_pad + n_frames, segment_hop)]) # [n_segments,]
            chordChange_segment = np.array([1] + [0 if x == y else 1 for x, y in zip(chord_segment[1:], chord_segment[:-1])])
            del chroma_TC_segment

            """BillboardData_segment = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array}, ...}"""
            BillboardData_shift_segment[key] = {}
            BillboardData_shift_segment[key]['chroma'] = chroma_segment.astype(np.float32)
            BillboardData_shift_segment[key]['TC'] = TC_segment.astype(np.float32)
            BillboardData_shift_segment[key]['chord'] = chord_segment.astype(np.int32)
            BillboardData_shift_segment[key]['chordChange'] = chordChange_segment.astype(np.int32)

        # save the preprocessed data
        outputdir = 'preprocessed_data/Billboard_data_mirex_Mm_shift_segment_' + str(shift) + '.pkl'
        with open(outputdir, "wb") as output_file:
            pickle.dump(BillboardData_shift_segment, output_file)


def reshape_Billboard(n_steps=100):
    print('Running Message: reshape Billborad ...')

    for shift in range(12):
        inputdir = 'preprocessed_data/Billboard_data_mirex_Mm_shift_segment_' + str(shift) + '.pkl' # with segment
        with open(inputdir, 'rb') as input_data:
            BillboardData_shift = pickle.load(input_data)
        """BillboardData_shift = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array}, ...}"""

        BillboardData_reshape = {}
        for key, value in BillboardData_shift.items():
            chroma = value['chroma']
            TC = value['TC']
            chord = value['chord']
            chordChange = value['chordChange']

            n_frames = chroma.shape[0]
            n_pad = 0 if n_frames/n_steps == 0 else n_steps - (n_frames % n_steps)
            if n_pad != 0: # chek if need paddings
                chroma = np.pad(chroma, [(0, n_pad), (0, 0)], 'constant', constant_values=0)
                TC = np.pad(TC, [(0, n_pad), (0, 0)], 'constant', constant_values=0)
                chord = np.pad(chord, [(0, n_pad)], 'constant', constant_values=24) # 24 for padding frams
                chordChange = np.pad(chordChange, [(0, n_pad)], 'constant', constant_values=0) # 0 for padding frames

            seq_hop = n_steps // 2
            n_sequences = int((chroma.shape[0] - n_steps) / seq_hop) + 1
            _, feature_size = chroma.shape
            _, TC_size = TC.shape
            s0, s1 = chroma.strides
            chroma_reshape = np.lib.stride_tricks.as_strided(chroma, shape=(n_sequences, n_steps, feature_size), strides=(s0 * seq_hop, s0, s1))
            ss0, ss1 = TC.strides
            TC_reshape = np.lib.stride_tricks.as_strided(TC, shape=(n_sequences, n_steps, TC_size), strides=(ss0 * seq_hop, ss0, ss1))
            sss0, = chord.strides
            chord_reshape = np.lib.stride_tricks.as_strided(chord, shape=(n_sequences, n_steps), strides=(sss0 * seq_hop, sss0))
            ssss0, = chordChange.strides
            chordChange_reshape = np.lib.stride_tricks.as_strided(chordChange, shape=(n_sequences, n_steps), strides=(ssss0 * seq_hop, ssss0))
            sequenceLen = np.array([n_steps for _ in range(n_sequences - 1)] + [n_steps - n_pad], dtype=np.int32) # [n_sequences]

            """BillboardData_reshape = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array, 'sequenceLen': array, 'nSequence': array}, ...}"""
            BillboardData_reshape[key] = {}
            BillboardData_reshape[key]['chroma'] = chroma_reshape
            BillboardData_reshape[key]['TC'] = TC_reshape
            BillboardData_reshape[key]['chord'] = chord_reshape
            BillboardData_reshape[key]['chordChange'] = chordChange_reshape
            BillboardData_reshape[key]['sequenceLen'] = sequenceLen
            BillboardData_reshape[key]['nSequence'] = n_sequences
            # BillboardData_reshape[key]['nSegment'] = (n_sequences // 2 + 1)*n_steps - n_pad

        # save the preprocessed data
        outputdir = 'preprocessed_data/Billboard_data_mirex_Mm_reshape_' + str(shift) + '.pkl' # with segment
        with open(outputdir, "wb") as output_file:
            pickle.dump(BillboardData_reshape, output_file)


def split_dataset():
    print('Running Message: split dataset into training, validation sets ...')

    # inputdir = 'preprocessed_data\\sets.pkl'
    # with open(inputdir, 'rb') as input_file:
    #     train_set, test_set = pickle.load(input_file)
    #     print(train_set)
    #     print(test_set)

    inputdir = 'preprocessed_data/Billboard_data_mirex_Mm_reshape_0.pkl' # with segment
    with open(inputdir, 'rb') as input_data:
        BillboardData_reshape_0 = pickle.load(input_data)
        """BillboardData_reshape = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array, 'sequenceLen': array, 'nSequence': array}, ...}"""

    # split dataset into training, validation, and testing sets
    dir = "billboard/chordino"
    ops = sorted(os.listdir(dir))
    split_sets = {'train':[], 'valid': []}
    for op in ops:
        if int(op) < 1000:
            info = (op, BillboardData_reshape_0[op]['nSequence'] // 2 + 1)  # (opus, number of sequences)
            split_sets['train'].append(info)
        else:
            info = (op, BillboardData_reshape_0[op]['nSequence'] // 2 + 1)  # (opus, number of sequences)
            split_sets['valid'].append(info)

    x_valid = np.concatenate([BillboardData_reshape_0[info[0]]['chroma'][::2] for info in split_sets['valid']], axis=0)
    TC_valid = np.concatenate([BillboardData_reshape_0[info[0]]['TC'][::2] for info in split_sets['valid']], axis=0)
    y_valid = np.concatenate([BillboardData_reshape_0[info[0]]['chord'][::2] for info in split_sets['valid']], axis=0)
    y_cc_valid = np.concatenate([BillboardData_reshape_0[info[0]]['chordChange'][::2] for info in split_sets['valid']], axis=0)
    y_len_valid = np.concatenate([BillboardData_reshape_0[info[0]]['sequenceLen'][::2] for info in split_sets['valid']], axis=0)
    del BillboardData_reshape_0

    x_train, TC_train, y_train, y_cc_train, y_len_train = [], [], [], [], []
    for shift in range(12):
        inputdir = 'preprocessed_data\\Billboard_data_mirex_Mm_reshape_' + str(shift) + '.pkl' # with segment
        with open(inputdir, 'rb') as input_data:
            BillboardData_reshape = pickle.load(input_data)
            """BillboardData_reshape = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array, 'sequenceLen': array, 'nSequence': array}, ...}"""
            x_train.append(np.concatenate([BillboardData_reshape[info[0]]['chroma'][::2] for info in split_sets['train']], axis=0))
            TC_train.append(np.concatenate([BillboardData_reshape[info[0]]['TC'][::2] for info in split_sets['train']], axis=0))
            y_train.append(np.concatenate([BillboardData_reshape[info[0]]['chord'][::2] for info in split_sets['train']], axis=0))
            y_cc_train.append(np.concatenate([BillboardData_reshape[info[0]]['chordChange'][::2] for info in split_sets['train']], axis=0))
            y_len_train.append(np.concatenate([BillboardData_reshape[info[0]]['sequenceLen'][::2] for info in split_sets['train']], axis=0))
            del BillboardData_reshape
    x_train = np.concatenate(x_train, axis=0)
    TC_train = np.concatenate(TC_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_cc_train = np.concatenate(y_cc_train, axis=0)
    y_len_train  = np.concatenate(y_len_train, axis=0)

    outputdir = 'preprocessed_data/Billboard_data_mirex_Mm_model_input_final.npz' # with segment
    with open(outputdir, "wb") as output_file:
        np.savez_compressed(output_file,
                            x_train=x_train,
                            TC_train=TC_train,
                            y_train=y_train,
                            y_cc_train=y_cc_train,
                            y_len_train=y_len_train,
                            x_valid=x_valid,
                            TC_valid=TC_valid,
                            y_valid=y_valid,
                            y_cc_valid=y_cc_valid,
                            y_len_valid=y_len_valid,
                            split_sets=split_sets)

    print('preprocessing finished')


if __name__ == '__main__':
    load_billboard('./billboard')
    augment_Billboard()
    segment_Billboard()
    reshape_Billboard()
    split_dataset()