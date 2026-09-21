from guard_monitoring.storage import SQLiteJobStore


def test_sqlite_job_store_persists(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    store = SQLiteJobStore(db)
    created = store.create("job1", "input.mp4", "out/job1")
    assert created.status == "queued"

    store.update("job1", status="completed", progress=100.0, summary={"frames": 4})

    reopened = SQLiteJobStore(db)
    job = reopened.get("job1")
    assert job is not None
    assert job.status == "completed"
    assert job.summary == {"frames": 4}
