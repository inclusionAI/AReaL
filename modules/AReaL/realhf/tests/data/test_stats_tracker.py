import pytest
import torch

from realhf.base.stats_tracker import DistributedStatsTracker, ReduceType


@pytest.fixture
def tracker():
    return DistributedStatsTracker()


def test_basic_stat_recording(tracker):
    # Test basic stat recording and averaging
    mask = torch.BoolTensor([True, False, True])
    values = torch.FloatTensor([1.0, 2.0, 3.0])

    tracker.denominator(mask=mask)
    tracker.stat(denominator="mask", value=values)

    results = tracker.export()
    assert pytest.approx(results["value/avg"]) == 2.0  # (1+3)/2
    assert pytest.approx(results["value/min"]) == 1.0
    assert pytest.approx(results["value/max"]) == 3.0


def test_scoping(tracker):
    # Test hierarchical scoping
    with tracker.scope("parent"):
        tracker.denominator(parent_mask=torch.BoolTensor([True]))
        tracker.stat(parent_value=torch.FloatTensor([1.0]), denominator="parent_mask")
        with tracker.scope("child"):
            with pytest.raises(ValueError):
                tracker.stat(denominator="child_mask", value=torch.FloatTensor([1.0]))
            tracker.denominator(child_mask=torch.BoolTensor([1.0]))
            tracker.stat(denominator="child_mask", value=torch.FloatTensor([1.0]))

    results = tracker.export()
    assert "parent/parent_mask" in results
    assert "parent/parent_value/avg" in results
    assert "parent/child/child_mask" in results
    assert "parent/child/value/avg" in results


def test_reduce_types(tracker):
    # Test different reduce types
    mask = torch.BoolTensor([True, False, True])
    values = torch.FloatTensor([1.0, 2.0, 3.0])

    tracker.denominator(mask=mask)
    tracker.stat(denominator="mask", reduce_type=ReduceType.SUM, sum_val=values)
    tracker.stat(denominator="mask", reduce_type=ReduceType.MIN, min_val=values)
    tracker.stat(denominator="mask", reduce_type=ReduceType.MAX, max_val=values)

    results = tracker.export()
    assert pytest.approx(results["sum_val"]) == 4.0  # 1+3
    assert pytest.approx(results["min_val"]) == 1.0
    assert pytest.approx(results["max_val"]) == 3.0


def test_validation_checks(tracker):
    # Test input validation
    with pytest.raises(ValueError):
        tracker.denominator(invalid=torch.FloatTensor([1.0]))  # Not bool tensor

    tracker.denominator(mask=torch.BoolTensor([True]))
    with pytest.raises(ValueError):
        tracker.stat(denominator="nonexistent", value=torch.FloatTensor([1.0]))

    with pytest.raises(AssertionError):
        tracker.stat(
            denominator="mask", value=torch.FloatTensor([1.0, 2.0])  # Shape mismatch
        )


def test_multiple_recordings(tracker):
    # Test multiple recordings
    mask1 = torch.BoolTensor([True, False])
    mask2 = torch.BoolTensor([False, True])
    values1 = torch.FloatTensor([1.0, 2.0])
    values2 = torch.FloatTensor([3.0, 4.0])

    tracker.denominator(mask=mask1)
    tracker.denominator(mask=mask2)
    tracker.stat(denominator="mask", value=values1)
    tracker.stat(denominator="mask", value=values2)

    results = tracker.export()
    assert (
        pytest.approx(results["value/avg"]) == (1.0 + 4.0) / 2
    )  # (1 from 1st, 4 from 2nd)


def test_denominator_edge_cases(tracker):
    # Test edge cases with denominators
    with pytest.raises(ValueError):  # Should fail on shape check
        empty_mask = torch.BoolTensor([])
        tracker.denominator(mask=empty_mask)

    zero_mask = torch.BoolTensor([False, False])
    tracker.denominator(mask=zero_mask)
    tracker.stat(denominator="mask", value=torch.FloatTensor([1.0, 2.0]))
    results = tracker.export()
    assert "value/min" not in results
    assert "value/max" not in results
    assert "value/avg" not in results


def test_key_specific_export(tracker):
    # Test exporting specific keys
    tracker.denominator(mask1=torch.BoolTensor([True]), mask2=torch.BoolTensor([True]))
    tracker.stat(denominator="mask1", value1=torch.FloatTensor([1.0]))
    tracker.stat(denominator="mask2", value2=torch.FloatTensor([2.0]))

    result = tracker.export(key="value1")
    assert "value1/avg" in result
    assert "value2/avg" not in result


def test_scalar_values(tracker):
    # Test scalar value recording and averaging
    tracker.scalar(scalar1=1.0, scalar2=2.0)
    tracker.scalar(scalar1=3.0, scalar2=4.0)

    results = tracker.export()
    assert pytest.approx(results["scalar1"]) == 2.0  # (1+3)/2
    assert pytest.approx(results["scalar2"]) == 3.0  # (2+4)/2


def test_moe_aux_losses(monkeypatch, tracker):
    # Test MOE auxiliary losses handling
    from realhf.base.stats_tracker import MOE_AUX_LOSSES

    # Mock distributed environment
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.all_reduce", lambda x, group: x)

    # Mock pipe parallel group and last stage check
    mock_group = object()
    monkeypatch.setattr("realhf.base.constants.pipe_parallel_group", lambda: mock_group)
    monkeypatch.setattr("realhf.base.constants.is_last_pipe_stage", lambda: True)

    # Set up test MOE losses
    MOE_AUX_LOSSES["moe_loss1"] = torch.tensor([1.0, 2.0])
    MOE_AUX_LOSSES["moe_loss2"] = torch.tensor([3.0, 4.0])

    results = tracker.export()
    assert pytest.approx(results["moe_loss1"]) == 1.5  # (1+2)/2
    assert pytest.approx(results["moe_loss2"]) == 3.5  # (3+4)/2
    assert not MOE_AUX_LOSSES  # Should be cleared after export


def test_empty_tracker(tracker):
    # Test exporting from an empty tracker
    results = tracker.export()
    assert results == {}


def test_reset_behavior(tracker):
    # Test that stats are reset after export
    tracker.denominator(mask=torch.BoolTensor([True]))
    tracker.stat(denominator="mask", value=torch.FloatTensor([1.0]))

    results1 = tracker.export()
    assert "value/avg" in results1

    results2 = tracker.export()
    assert results2 == {}


def test_no_reset_behavior(tracker):
    # Test that stats are preserved when reset=False
    tracker.denominator(mask=torch.BoolTensor([True]))
    tracker.stat(denominator="mask", value=torch.FloatTensor([1.0]))

    results1 = tracker.export(reset=False)
    assert "value/avg" in results1

    results2 = tracker.export()
    assert "value/avg" in results2  # Should still be there


def test_default_tracker():
    # Test the default tracker instance and its functions
    mask = torch.BoolTensor([True, False, True])
    values = torch.FloatTensor([1.0, 2.0, 3.0])

    from realhf.base.stats_tracker import denominator, export, stat

    denominator(mask=mask)
    stat(denominator="mask", value=values)

    results = export()
    assert pytest.approx(results["value/avg"]) == 2.0


def test_reduce_type_validation(tracker):
    # Test invalid reduce type handling
    with pytest.raises(ValueError):
        tracker._set_reduce_type("key", "invalid_type")  # Not a ReduceType enum

    with pytest.raises(ValueError):
        tracker.denominator(mask=torch.BoolTensor([True]))
        tracker.stat(
            denominator="mask", reduce_type="invalid", value=torch.FloatTensor([1.0])
        )


def test_scalar_reduce_type_validation(tracker):
    # Test that SCALAR reduce type can't be used with tensors
    tracker.denominator(mask=torch.BoolTensor([True]))
    with pytest.raises(ValueError):
        tracker.stat(
            denominator="mask",
            reduce_type=ReduceType.SCALAR,
            value=torch.FloatTensor([1.0]),
        )


def test_full_key_generation(tracker):
    # Test full key generation with and without scope
    assert tracker._get_full_key("key") == "key"

    with tracker.scope("scope1"):
        assert tracker._get_full_key("key") == "scope1/key"

        with tracker.scope("scope2"):
            assert tracker._get_full_key("key") == "scope1/scope2/key"

    # Test with empty name in constructor
    empty_tracker = DistributedStatsTracker(name="")
    assert empty_tracker._get_full_key("key") == "key"

    # Test with name in constructor
    named_tracker = DistributedStatsTracker(name="root")
    assert named_tracker._get_full_key("key") == "root/key"
