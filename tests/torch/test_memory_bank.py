import pytest
import torch

from drifting_torch.memory_bank import ClassMemoryBank


def test_memory_bank_ring_buffer_and_initialized_sampling():
    bank = ClassMemoryBank(num_classes=2, capacity=3, seed=19)
    bank.add(torch.tensor([[1.0], [2.0], [3.0], [4.0]]), torch.tensor([0, 0, 0, 0]))
    assert bank.count.tolist() == [3, 0]
    assert bank.pointer.tolist() == [1, 0]
    assert sorted(bank.storage[0, :, 0].tolist()) == [2.0, 3.0, 4.0]
    sampled = bank.sample(torch.tensor([0]), 20)
    assert sampled.shape == (1, 20, 1)
    assert set(sampled.flatten().tolist()) <= {2.0, 3.0, 4.0}


def test_memory_bank_round_trip_preserves_rng_sequence():
    bank = ClassMemoryBank(num_classes=2, capacity=4, seed=23)
    bank.add(torch.arange(8, dtype=torch.float32).reshape(8, 1), torch.arange(8) % 2)
    restored = ClassMemoryBank.from_state_dict(bank.state_dict())
    assert restored.state_dict_equal(bank)
    labels = torch.tensor([0, 1, 1, 0])
    assert torch.equal(bank.sample(labels, 3), restored.sample(labels, 3))


def test_memory_bank_validates_labels_shape_dtype_and_empty_classes():
    bank = ClassMemoryBank(num_classes=2, capacity=2)
    with pytest.raises(RuntimeError, match="empty"):
        bank.sample(torch.tensor([0]), 1)
    with pytest.raises(ValueError, match="integer"):
        bank.add(torch.ones(1, 2), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="range"):
        bank.add(torch.ones(1, 2), torch.tensor([2]))
    bank.add(torch.ones(1, 2), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="class 1"):
        bank.sample(torch.tensor([1]), 1)

