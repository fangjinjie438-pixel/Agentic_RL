"""Correct worker padding for shrinking active batches in the pinned backend."""

from verl import DataProto


def pad_for_workers(data, world_size):
    """Repeat rows until divisible, including when len(data) < world_size / 2.

The bundled pad_dataproto_to_divisor appends a single data[:pad_size], which
cannot add enough rows when only one trajectory remains on three GPUs.
    """
    if world_size < 1 or len(data) < 1:
        raise ValueError("Worker padding requires a nonempty batch and positive world size")
    padding = (-len(data)) % world_size
    if not padding:
        return data, 0
    chunks, remaining = [data], padding
    while remaining:
        take = min(remaining, len(data))
        chunks.append(data[:take])
        remaining -= take
    return DataProto.concat(chunks), padding
