import torch

from drifting_torch.distributed.collectives import gather_valid
from drifting_torch.distributed.runtime import DistributedContext


def test_gather_valid_masks_single_process():
    context = DistributedContext.single()
    values, mask = gather_valid(
        torch.tensor([[1.0], [99.0], [2.0]]),
        torch.tensor([True, False, True]),
        context,
    )
    torch.testing.assert_close(values, torch.tensor([[1.0], [2.0]]))
    assert mask.tolist() == [True, True]
