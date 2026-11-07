from memrag.infrastructure.embedding.probe import _MemSampler


def test_sampler_records_a_memory_drop_and_stops():
    sampler = _MemSampler(interval_s=0.005)
    sampler.start()
    block = bytearray(64 * 2**20)  # keep 64 MiB alive while sampling
    for i in range(0, len(block), 4096):
        block[i] = 1
    drop = sampler.stop()
    assert not sampler.is_alive()
    assert drop >= 0
    del block
