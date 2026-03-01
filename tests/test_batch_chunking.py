from app import _chunk_by_char_budget, _rebalance_chunks_for_parallelism


def test_rebalance_chunks_increases_chunk_count_preserving_order() -> None:
    chunks = [["a", "b", "c", "d"]]

    out = _rebalance_chunks_for_parallelism(chunks, desired_chunks=3)

    assert out == [["a", "b"], ["c"], ["d"]]


def test_rebalance_chunks_does_not_exceed_item_count() -> None:
    chunks = [["a", "b"]]

    out = _rebalance_chunks_for_parallelism(chunks, desired_chunks=8)

    assert out == [["a"], ["b"]]


def test_chunk_by_char_budget_respects_max_items() -> None:
    texts = [str(i) for i in range(10)]

    chunks = _chunk_by_char_budget(texts, target_chars=1000, max_items=3)

    assert [len(c) for c in chunks] == [3, 3, 3, 1]
