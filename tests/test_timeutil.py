import pytest

from areal.utils.timeutil import EpochStepTimeFreqCtl


def test_start_epoch_zero_triggers_initial_check_once():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=0,
    )

    assert ctl.check(epochs=0, steps=0) is True
    assert ctl.check(epochs=0, steps=0) is False


def test_start_epoch_one_matches_legacy_epoch_end_trigger():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=1,
    )

    # No epoch boundary yet.
    assert ctl.check(epochs=0, steps=0) is False
    # First epoch end triggers.
    assert ctl.check(epochs=1, steps=0) is True


def test_start_epoch_three_freq_one():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=3,
    )

    assert ctl.check(epochs=1, steps=0) is False
    assert ctl.check(epochs=1, steps=0) is False
    assert ctl.check(epochs=1, steps=0) is True
    assert ctl.check(epochs=1, steps=0) is True


def test_start_epoch_two_freq_two():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=2,
        freq_step=None,
        freq_sec=None,
        start_epoch=2,
    )

    # Epoch 1: before start.
    assert ctl.check(epochs=1, steps=0) is False
    # Epoch 2: first trigger.
    assert ctl.check(epochs=1, steps=0) is True
    # Epoch 3: not on interval.
    assert ctl.check(epochs=1, steps=0) is False
    # Epoch 4: second trigger.
    assert ctl.check(epochs=1, steps=0) is True


def test_state_dict_roundtrip_keeps_schedule_progress():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=2,
        freq_step=None,
        freq_sec=None,
        start_epoch=2,
    )
    assert ctl.check(epochs=1, steps=0) is False  # epoch_count=1

    state = ctl.state_dict()
    recovered = EpochStepTimeFreqCtl(
        freq_epoch=2,
        freq_step=None,
        freq_sec=None,
        start_epoch=2,
    )
    recovered.load_state_dict(state)

    # epoch_count goes 1->2 and should trigger.
    assert recovered.check(epochs=1, steps=0) is True


def test_load_state_dict_restores_start_epoch():
    source = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=3,
    )
    state = source.state_dict()

    target = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=1,
    )
    target.load_state_dict(state)

    assert target.start_epoch == 3


def test_load_state_dict_supports_missing_new_fields():
    ctl = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=0,
    )

    state = ctl.state_dict()
    # Simulate older checkpoint payload without newly added fields.
    state.pop("start_epoch")
    state.pop("epoch_count")

    recovered = EpochStepTimeFreqCtl(
        freq_epoch=1,
        freq_step=None,
        freq_sec=None,
        start_epoch=1,
    )
    recovered.load_state_dict(state)

    # Missing fields should fall back to runtime defaults.
    assert recovered.start_epoch == 1


def test_start_epoch_negative_raises():
    with pytest.raises(ValueError):
        EpochStepTimeFreqCtl(
            freq_epoch=1,
            freq_step=None,
            freq_sec=None,
            start_epoch=-1,
        )
