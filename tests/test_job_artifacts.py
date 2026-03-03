from __future__ import annotations

from job_artifacts import should_checkpoint


def test_should_checkpoint_on_completion_even_if_interval_not_met() -> None:
    assert should_checkpoint(batch_index=3, total_batches=3, checkpoint_every_batches=10) is True


def test_should_checkpoint_respects_interval_during_progress() -> None:
    assert should_checkpoint(batch_index=1, total_batches=10, checkpoint_every_batches=5) is False
    assert should_checkpoint(batch_index=5, total_batches=10, checkpoint_every_batches=5) is True
