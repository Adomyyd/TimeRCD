import random

import pytest

from src.data import LengthBucketBatchSampler


BATCH_SIZE = 32


def synthetic_lengths(samples: int = 200000, seed: int = 0) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    lengths = [rng.randint(1000, 10000) for _ in range(samples)]
    return list(range(samples)), lengths


def padded_fraction(batches, lengths) -> float:
    total = sum(len(batch) * max(lengths[index] for index in batch) for batch in batches)
    useful = sum(sum(lengths[index] for index in batch) for batch in batches)
    return 1.0 - useful / total


def test_every_index_is_used_once_per_epoch():
    indices, lengths = synthetic_lengths()
    sampler = LengthBucketBatchSampler(indices, lengths, batch_size=BATCH_SIZE, seed=72)
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == indices
    assert len(sampler) == len(batches)
    assert all(len(batch) == BATCH_SIZE for batch in batches[:-1])
    assert 0 < len(batches[-1]) <= BATCH_SIZE


def test_bucketing_keeps_step_count_and_cuts_padding():
    indices, lengths = synthetic_lengths()
    sequential = [indices[start:start + BATCH_SIZE] for start in range(0, len(indices), BATCH_SIZE)]
    bucketed = list(LengthBucketBatchSampler(indices, lengths, batch_size=BATCH_SIZE, seed=72))
    assert len(bucketed) == len(sequential)
    assert padded_fraction(sequential, lengths) > 0.35
    assert padded_fraction(bucketed, lengths) < 0.08


def test_same_seed_and_epoch_is_reproducible():
    indices, lengths = synthetic_lengths(samples=4000)
    first = LengthBucketBatchSampler(indices, lengths, BATCH_SIZE, seed=72)
    second = LengthBucketBatchSampler(indices, lengths, BATCH_SIZE, seed=72)
    first.set_epoch(3)
    second.set_epoch(3)
    assert list(first) == list(second)
    first.set_epoch(4)
    assert list(first) != list(second)


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0},
    {"batch_size": BATCH_SIZE, "bucket_multiplier": 0},
])
def test_invalid_settings_fail_fast(kwargs):
    indices, lengths = synthetic_lengths(samples=64)
    with pytest.raises(ValueError):
        LengthBucketBatchSampler(indices, lengths, seed=72, **kwargs)


def test_mismatched_lengths_fail_fast():
    with pytest.raises(ValueError):
        LengthBucketBatchSampler([1, 2, 3], [10, 20], batch_size=2, seed=72)
