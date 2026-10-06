from guard_monitoring.config import load_config


def test_test_mode_preserves_models_and_uses_short_durations(monkeypatch):
    monkeypatch.setenv("GUARD_TEST_MODE", "true")
    monkeypatch.setenv("GUARD_TEST_SLEEP_SECONDS", "4")
    cfg = load_config(source="x.mp4", output_dir="out")
    assert cfg["rules"]["sleep_seconds"] == 4.0
    assert cfg["rules"]["phone_seconds"] == 5.0
    assert cfg["models"]["guard_detector"]["size"] == "medium"
    assert cfg["models"]["guard_detector"]["require_finetuned"] is True


def test_test_mode_posture_duration_respects_explicit_override(monkeypatch):
    monkeypatch.setenv("GUARD_TEST_MODE", "true")
    monkeypatch.delenv("GUARD_SLEEP_FALLBACK_SECONDS", raising=False)
    monkeypatch.setenv("GUARD_TEST_SLEEP_FALLBACK_SECONDS", "4")
    assert load_config()["sleep_logic"]["fallback_confirm_seconds"] == 4
    monkeypatch.setenv("GUARD_SLEEP_FALLBACK_SECONDS", "7")
    assert load_config()["sleep_logic"]["fallback_confirm_seconds"] == 7
    monkeypatch.setenv("GUARD_TEST_MODE", "false")
    monkeypatch.delenv("GUARD_SLEEP_FALLBACK_SECONDS")
    assert load_config()["sleep_logic"]["fallback_confirm_seconds"] == 8


def test_invalid_zone_fails_fast(monkeypatch):
    monkeypatch.setenv("GUARD_DUTY_ZONE_JSON", "[[0,0],[1,0]]")
    try:
        load_config(source="x.mp4", output_dir="out")
    except ValueError as exc:
        assert "GUARD_DUTY_ZONE_JSON" in str(exc)
    else:
        raise AssertionError("invalid duty zone should fail")
