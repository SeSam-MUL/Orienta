"""Tests for ebsd_ai.sync.server_sync — network drive synchronization."""

from __future__ import annotations

import h5py
import pytest

from ebsd_ai.sync.server_sync import ServerSync, SyncResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shard(directory, name: str = "shard_1000_abc.h5", n_keys: int = 1):
    """Create a minimal HDF5 shard file in *directory*."""
    path = directory / name
    with h5py.File(path, "w") as f:
        f.attrs["created"] = 1000
        for i in range(n_keys):
            f.create_group(f"sample_{i:08d}")
    return path


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------

class TestSyncResult:
    def test_defaults(self):
        r = SyncResult()
        assert r.n_copied == 0
        assert r.n_skipped == 0
        assert r.n_failed == 0
        assert r.success is True
        assert r.elapsed_seconds == 0.0

    def test_counts(self):
        r = SyncResult(
            files_copied=["a.h5", "b.h5"],
            files_skipped=["c.h5"],
            files_failed=["d.h5"],
            errors={"d.h5": "oops"},
        )
        assert r.n_copied == 2
        assert r.n_skipped == 1
        assert r.n_failed == 1
        assert r.success is False

    def test_success_with_skips(self):
        r = SyncResult(files_skipped=["a.h5"])
        assert r.success is True


# ---------------------------------------------------------------------------
# ServerSync init
# ---------------------------------------------------------------------------

class TestServerSyncInit:
    def test_missing_local_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Local directory"):
            ServerSync(
                local_dir=tmp_path / "nonexistent",
                server_dir=tmp_path / "server",
            )

    def test_valid_init(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        sync = ServerSync(local_dir=local, server_dir=tmp_path / "server")
        assert sync.local_dir == local
        assert sync.server_dir == tmp_path / "server"


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

class TestPush:
    def test_push_creates_server_dir(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        _make_shard(local, "shard_100_aaa.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert server.is_dir()
        assert result.n_copied == 1
        assert result.success is True

    def test_push_copies_shard(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        _make_shard(local, "shard_100_aaa.h5", n_keys=3)

        sync = ServerSync(local_dir=local, server_dir=server)
        sync.push()

        dest = server / "shard_100_aaa.h5"
        assert dest.exists()
        with h5py.File(dest, "r") as f:
            assert len(f.keys()) == 3

    def test_push_skips_existing(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(server, "shard_100_aaa.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert result.n_copied == 0
        assert result.n_skipped == 1
        assert result.success is True

    def test_push_multiple_shards(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(local, "shard_200_bbb.h5")
        _make_shard(local, "shard_300_ccc.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert result.n_copied == 3
        assert result.success is True

    def test_push_mixed_skip_and_copy(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(local, "shard_200_bbb.h5")
        _make_shard(server, "shard_100_aaa.h5")  # already on server

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert result.n_copied == 1
        assert result.n_skipped == 1
        assert "shard_200_bbb.h5" in result.files_copied
        assert "shard_100_aaa.h5" in result.files_skipped

    def test_push_empty_local(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert result.n_copied == 0
        assert result.n_skipped == 0
        assert result.success is True

    def test_push_elapsed_seconds(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        _make_shard(local, "shard_100_aaa.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

class TestPull:
    def test_pull_copies_from_server(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(server, "shard_500_xyz.h5", n_keys=2)

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.pull()

        assert result.n_copied == 1
        assert (local / "shard_500_xyz.h5").exists()

    def test_pull_skips_existing(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(local, "shard_500_xyz.h5")
        _make_shard(server, "shard_500_xyz.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.pull()

        assert result.n_copied == 0
        assert result.n_skipped == 1

    def test_pull_empty_server(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.pull()

        assert result.n_copied == 0
        assert result.success is True

    def test_pull_nonexistent_server(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"  # not created

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.pull()

        # No server dir → nothing to pull, but no error
        assert result.n_copied == 0
        assert result.success is True


# ---------------------------------------------------------------------------
# Full sync (bidirectional)
# ---------------------------------------------------------------------------

class TestFullSync:
    def test_full_sync_bidirectional(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(server, "shard_200_bbb.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        push_result, pull_result = sync.full_sync()

        # local -> server: shard_100_aaa copied
        assert push_result.n_copied == 1
        assert "shard_100_aaa.h5" in push_result.files_copied

        # server -> local: shard_200_bbb copied
        assert pull_result.n_copied == 1
        assert "shard_200_bbb.h5" in pull_result.files_copied

        # Both dirs now have both files
        assert (server / "shard_100_aaa.h5").exists()
        assert (local / "shard_200_bbb.h5").exists()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_push_callback_called(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(local, "shard_200_bbb.h5")

        calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, fname: str) -> None:
            calls.append((current, total, fname))

        sync = ServerSync(local_dir=local, server_dir=server)
        sync.push(progress_callback=on_progress)

        assert len(calls) == 2
        assert calls[0][1] == 2  # total is 2
        assert calls[1][0] == 2  # second call: current=2

    def test_pull_callback_called(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        server.mkdir()

        _make_shard(server, "shard_300_ccc.h5")

        calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, fname: str) -> None:
            calls.append((current, total, fname))

        sync = ServerSync(local_dir=local, server_dir=server)
        sync.pull(progress_callback=on_progress)

        assert len(calls) == 1
        assert calls[0] == (1, 1, "shard_300_ccc.h5")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_local_stats(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        _make_shard(local, "shard_100_aaa.h5")
        _make_shard(local, "shard_200_bbb.h5")

        sync = ServerSync(local_dir=local, server_dir=tmp_path / "server")
        stats = sync.local_stats()

        assert stats["n_shards"] == 2
        assert stats["total_size_bytes"] > 0

    def test_server_stats_empty(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"
        # server doesn't exist yet

        sync = ServerSync(local_dir=local, server_dir=server)
        stats = sync.server_stats()

        assert stats["n_shards"] == 0
        assert stats["total_size_bytes"] == 0

    def test_server_stats_after_push(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        _make_shard(local, "shard_100_aaa.h5", n_keys=5)

        sync = ServerSync(local_dir=local, server_dir=server)
        sync.push()

        stats = sync.server_stats()
        assert stats["n_shards"] == 1
        assert stats["total_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Atomic copy / failure handling
# ---------------------------------------------------------------------------

class TestAtomicCopy:
    def test_no_temp_files_left_on_success(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        _make_shard(local, "shard_100_aaa.h5")

        sync = ServerSync(local_dir=local, server_dir=server)
        sync.push()

        # No .tmp files should remain
        tmp_files = list(server.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_only_shard_files_are_synced(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        server = tmp_path / "server"

        # Create a shard and a non-shard file
        _make_shard(local, "shard_100_aaa.h5")
        (local / "other_file.txt").write_text("not a shard")
        (local / "data.h5").write_text("also not a shard pattern")

        sync = ServerSync(local_dir=local, server_dir=server)
        result = sync.push()

        # Only the shard file should be copied
        assert result.n_copied == 1
        assert not (server / "other_file.txt").exists()
        assert not (server / "data.h5").exists()


# ---------------------------------------------------------------------------
# Integration with TrainingStore
# ---------------------------------------------------------------------------

class TestTrainingStoreIntegration:
    def test_sync_real_store_shards(self, tmp_path):
        """Push/pull works with actual TrainingStore-generated shards."""
        import numpy as np
        from ebsd_ai.config import DetectorInfo
        from ebsd_ai.data.training_store import TrainingStore

        # Create a store and add some samples
        local = tmp_path / "local"
        store = TrainingStore(local_path=local, samples_per_shard=5)

        det = DetectorInfo()
        for i in range(3):
            store.add_sample(
                pattern=np.random.randint(0, 255, (60, 80), dtype=np.uint8),
                confirmed_phase=f"Phase_{i % 2}",
                detector_info=det,
                eds_data={"Fe": 70.0, "Ni": 10.0} if i % 2 == 0 else None,
            )

        # Now sync to server
        server = tmp_path / "server"
        sync = ServerSync(local_dir=local, server_dir=server)
        push_result = sync.push()

        assert push_result.n_copied >= 1
        assert push_result.success is True

        # Server can be loaded as a store
        server_store = TrainingStore(local_path=server)
        assert len(server_store) == 3

        # Pull back to a fresh local
        fresh_local = tmp_path / "fresh_local"
        fresh_local.mkdir()
        sync2 = ServerSync(local_dir=fresh_local, server_dir=server)
        pull_result = sync2.pull()

        assert pull_result.n_copied >= 1
        fresh_store = TrainingStore(local_path=fresh_local)
        assert len(fresh_store) == 3
