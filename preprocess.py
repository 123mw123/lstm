import utils

DATA_DIR = "C:\\Users\Sai Teja\Desktop\ELL888-RNN\\CTC"
TRAIN_DIR = DATA_DIR + "\\TRAIN\\DR"
TEST_DIR = DATA_DIR + "\\TEST\\DR"
DEV_DIR = DATA_DIR + "\\TRAIN\\DR"

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
test_inputs = utils.read_audio_files(TEST_DIR)
   # test_inputs = utils.standardize_audios(test_inputs)
test_sequence_lengths = utils.get_sequence_lengths(test_inputs)
test_inputs = utils.make_sequences_same_length(test_inputs, test_sequence_lengths)