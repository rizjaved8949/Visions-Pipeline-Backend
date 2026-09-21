from guard_monitoring.config import load_config


def test_test_mode_changes_only_runtime_rule_durations(monkeypatch):
    monkeypatch.setenv("GUARD_TEST_MODE", "true")
    monkeypatch.setenv("GUARD_TEST_SLEEP_SECONDS", "4")
    cfg = load_config(source="x.mp4", output_dir="out")
    assert cfg["rules"]["sleep_seconds"] == 4.0
    assert cfg["rules"]["phone_seconds"] == 5.0
    assert cfg["models"]["guard_detector"]["size"] == "medium"
    assert cfg["models"]["guard_detector"]["require_finetuned"] is True


def test_invalid_zone_fails_fast(monkeypatch):
    monkeypatch.setenv("GUARD_DUTY_ZONE_JSON", "[[0,0],[1,0]]")
    try:
        load_config(source="x.mp4", output_dir="out")
    except ValueError as exc:
        assert "GUARD_DUTY_ZONE_JSON" in str(exc)
    else:
        raise AssertionError("invalid duty zone should fail")
