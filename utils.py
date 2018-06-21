from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import logging
import unicodedata
import codecs

import numpy as np
import scipy.io.wavfile as wav
#from python_speech_features import mfcc
from librosa.feature import mfcc
import librosa


def read_text_file(path):
    """
    Read text from file
    Args:
        path: string.
            Path to text file.
    Returns:
        string.
            Read text.
    """
    with codecs.open(path, encoding="utf-8") as file:
        return file.read()


def normalize_text(text, remove_apostrophe=True):
    """
    Normalize given text.
    Args:
        text: string.
            Given text.
        remove_apostrophe: bool.
            Whether to remove apostrophe in given text.
    Returns:
        string.
            Normalized text.
    """

    # Convert unicode characters to ASCII.
    result = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()

    # Remove apostrophes.
    if remove_apostrophe:
        result = result.replace("'", "")

    return re.sub("[^a-zA-Z']+", ' ', result).strip().lower()


def read_text_files(dir, extensions=['txt']):
    """
    Read text files.
    Args:
        dir: string.
            Data directory.
        extensions: list of strings.
            File extensions.
    Returns:
        files: array of texts.
    
    if not os.path.isdir(dir):
        logging.error("Text files directory %s is not found.", dir)
        return None

    if not all(isinstance(extension, str) for extension in extensions):
        logging.error("Variable 'extensions' is not a list of strings.")
        return None
    """
    # Get files list.
    a = []
    i = 1
    files_paths_list = []
    while(i<=8):
        print(i)
        directory = dir + str(i)
        for Dir in os.listdir(directory):
            for extension in extensions:
                file_glob = os.path.join(directory+'//'+Dir, '*.' + extension)
                files_paths_list.extend(glob.glob(file_glob))
        i = i+1

    # Read files.
    files = []
    for file_path in files_paths_list:
        file = read_text_file(file_path)
        file = normalize_text(file)
        files.append(file)

    files = np.array(files)
    print(files.shape)
    return files


def read_audio_files(dir, extensions=['wav']):
    """
    Read audio files.
    Args:
        dir: string.
            Data directory.
        extensions: list of strings.
            File extensions.
    Returns:
        files: array of audios.
    
    if not os.path.isdir(dir):
        logging.error("Audio files directory %s is not found.", dir)
        return None

    if not all(isinstance(extension, str) for extension in extensions):
        logging.error("Variable 'extensions' is not a list of strings.")
        return None
    """
    # Get files list.
    a = []
    i = 1
    files_paths_list = []
    while(i<=8):
        print(i)
        directory = dir + str(i)
        for Dir in os.listdir(directory):
            for extension in extensions:
                file_glob = os.path.join(directory+'\\'+Dir, '*.' + extension)
                files_paths_list.extend(glob.glob(file_glob))
        i = i+1

    # Read files.
    files = []
    for file_path in files_paths_list:
        audio_data,audio_rate  = librosa.load(file_path)#wav.read(file_path) 

        file = mfcc(audio_data, sr=audio_rate,n_mfcc= 13)
        files.append(file)
    print(len(files))
   # for i in files:
    #    print(i.shape)
    #files = np.array(files)
    
    return files


def make_char_array(text, space_token='<space>'):
    """
    Make text as char array. Replace spaces with space token.
    Args:
        text: string.
            Given text.
        space_token: string.
            Text which represents space char.
    Returns:
        string array.
            Split text.
    """
    result = np.hstack([space_token if x == ' ' else list(x) for x in text])
    return result


def sparse_tuples_from_sequences(sequences, dtype=np.int32):
    """
    Create a sparse representations of inputs.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indexes = []
    values = []

    for n, sequence in enumerate(sequences):
        indexes.extend(zip([n] * len(sequence), range(len(sequence))))
        values.extend(sequence)

    indexes = np.asarray(indexes, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indexes).max(0)[1] + 1], dtype=np.int64)

    return indexes, values, shape


def sequence_decoder(sequence, first_index=(ord('a') - 1)):
    """
    Read text files.
    Args:
        sequence: list of int.
            Encoded sequence
        first_index: int.
            First index (usually index of 'a').
    Returns:
        decoded_text: string.
    """
    decoded_text = ''.join([chr(x) for x in np.asarray(sequence) + first_index])
    # Replacing blank label to none.
    decoded_text = decoded_text.replace(chr(ord('z') + 1), '')
    # Replacing space label to space.
    decoded_text = decoded_text.replace(chr(ord('a') - 1), ' ')
    return decoded_text


def texts_encoder(texts, first_index=(ord('a') - 1), space_index=0, space_token='<space>'):
    """
    Encode texts to numbers.
    Args:
        texts: list of texts.
            Data directory.
        first_index: int.
            First index (usually index of 'a').
        space_index: int.
            Index of 'space'.
        space_token: string.
            'space' representation.
    Returns:
        array of encoded texts.
    """
    result = []
    for text in texts:
        item = make_char_array(text, space_token)
        item = np.asarray([space_index if x == space_token else ord(x) - first_index for x in item])
        result.append(item)
    result = np.array(result)
    np.save('test_labels',result)
    return result


def standardize_audios(inputs):
    """
    Standardize audio inputs.
    Args:
        inputs: array of audios.
            Audio files.
    Returns:
        decoded_text: array of audios.
    """
    result = []
    for i in range(len(inputs)):
        item = np.array((inputs[i] - np.mean(inputs[i])) / np.std(inputs[i]))
        result.append(item)
    print(len(result))
    return result


def get_sequence_lengths(inputs):
    """
    Get sequence length of each sequence.
    Args:
        inputs: list of lists where each element is a sequence.
    Returns:
        array of sequence lengths.
    """
    result = []
    for input in inputs:
        result.append(input.shape[1])
    result = np.array(result, dtype=np.int64)
    #np.save('test_sequence_lengths.npy',result)
    return result


def make_sequences_same_length(sequences, sequences_lengths, default_value=0.0):
    """
    Make sequences same length for avoiding value
    error: setting an array element with a sequence.
    Args:
        sequences: list of sequence arrays.
        sequences_lengths: list of int.
        default_value: float32.
            Default value of newly created array.
    Returns:
        result: array of with same dimensions [num_samples, max_length, num_features].
    """

    # Get number of sequnces.
    num_samples = len(sequences)
    #print(num_samples)
    max_length = np.max(sequences_lengths)
    #print(max_length)
    # Get shape of the first non-zero length sequence.
    sample_shape= tuple()
    final_samples = []
    num_features = sequences[0].shape[0]
    for s in sequences:
        #print(s.shape[1])
        if s.shape[1] < max_length:

            #print(sequences[0].shape[0])
            mat = np.zeros((s.shape[0],max_length))
            #print(mat.shape)
            mat[:s.shape[0],:s.shape[1] ] = s
            s = mat
        #print(s.shape)
        final_samples.append(s)


    #print(len(final_samples))
    final_samples = np.array(final_samples)
    final_samples = np.reshape(final_samples, (num_samples, max_length, num_features))
    print(final_samples.shape)
    
    return final_samples