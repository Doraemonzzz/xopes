try:
    from fla.ops.utils import chunk_local_cumsum
except:
    chunk_local_cumsum = lambda x: None


def chunk_local_cumsum_wrapper(x, dim, reverse, chunk_size):
    return chunk_local_cumsum(x, chunk_size, reverse=reverse, head_first=False)
