import torch
import allennlp
import json
import logging
import warnings
from typing import Any, Dict, List, Union, Callable
import numpy
import torch

from torch.nn.modules import Dropout

from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoCharacterMapper,
    ELMoTokenCharactersIndexer,
)
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import (
    add_sentence_boundary_token_ids,
    get_device_of,
    remove_sentence_boundaries,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


logger = logging.getLogger(__name__)


##########################################
# Sample code to use allennlp
##########################################

from allennlp.modules.elmo import Elmo, batch_to_ids

model_dir = 'D:/pretrained_model/elmo_original/'
options_file = model_dir + 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
weights_file = model_dir + 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

num_output_representations = 2  # 2或者1

elmo = Elmo(
    options_file=options_file,
    weight_file=weights_file,
    num_output_representations=num_output_representations,
    dropout=0
)

sentence_lists = [['I', 'have', 'a', 'dog', ',', 'it', 'is', 'so', 'cute'],
                  ['That', 'is', 'a', 'question'],
                  ['an']]

character_ids = batch_to_ids(sentence_lists)  #
print('character_ids:', character_ids.shape)  # [3, 11, 50]

res = elmo(character_ids)
print(len(res['elmo_representations']))  # 2
print(res['elmo_representations'][0].shape)  # [3, 9, 1024]
print(res['elmo_representations'][1].shape)  # [3, 9, 1024]

print(res['mask'].shape)  # [3, 9]


