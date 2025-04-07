import pytest

from realhf.api.core.model_api import FinetuneSpec, StepInfo


@pytest.mark.parametrize("total_train_epochs", [10])
@pytest.mark.parametrize("train_batch_size", [64, 25, 11, 38])
@pytest.mark.parametrize("dataset_size", [200, 168, 77])
def test_epoch_counter(
    total_train_epochs: int, train_batch_size: int, dataset_size: int
):
    ft_spec = FinetuneSpec(
        total_train_epochs=total_train_epochs,
        train_batch_size=train_batch_size,
        dataset_size=dataset_size,
    )
    version = StepInfo()
    _epoch = 0
    _epoch_step = 0
    _step = 0
    is_new_epoch_records = []
    is_last_step_records = []
    gt = []
    while _epoch < total_train_epochs:
        is_last_step_records.append(ft_spec.is_epoch_last_step(version))
        is_new_epoch = ft_spec.is_new_epoch(version)
        is_new_epoch_records.append(is_new_epoch)

        if is_new_epoch:
            version.epoch += 1
            version.epoch_step = 0

        assert version.epoch == _epoch
        assert version.epoch_step == _epoch_step
        assert version.global_step == _step

        version.epoch_step += 1
        version.global_step += 1

        _step += 1
        _epoch_step += 1
        if _step * train_batch_size >= dataset_size * (_epoch + 1):
            _epoch += 1
            _epoch_step = 0
            gt.append(True)
        else:
            gt.append(False)
    assert gt == is_last_step_records
    assert [False] + is_last_step_records[:-1] == is_new_epoch_records
