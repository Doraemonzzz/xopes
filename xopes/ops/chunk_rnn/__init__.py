import torch

from .parallel import chunk_rnn_parallel

# from .sequential import chunk_rnn_sequential

chunk_rnn_parallel_fn = chunk_rnn_parallel
chunk_rnn_sequential_fn = None
# chunk_rnn_sequential_fn = chunk_rnn_sequential
