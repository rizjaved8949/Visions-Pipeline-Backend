from guard_monitoring.monitoring.rules import TimedRuleEngine


def _config():
    return {
        "warmup_seconds": 0,
        "sleep_seconds": 5,
        "phone_seconds": 3,
        "stationary_seconds": 10,
        "absence_seconds": 2,
        "grace_seconds": {"sleep": 1, "phone": 1, "stationary": 1, "absence": 0},
    }


def test_phone_rule_triggers_once():
    engine = TimedRuleEngine("cam", _config())
    assert engine.update("phone", True, 0.0, 1, {}) is None
    assert engine.update("phone", True, 2.9, 1, {}) is None
    event = engine.update("phone", True, 3.0, 1, {})
    assert event is not None
    assert event.rule == "phone"
    assert engine.update("phone", True, 4.0, 1, {}) is None


def test_unknown_time_pauses_positive_timer():
    engine = TimedRuleEngine("cam", _config())
    assert engine.update("phone", True, 0.0, 1, {}) is None
    assert engine.update("phone", None, 1.0, 1, {}) is None
    assert engine.update("phone", None, 5.0, 1, {}) is None
    # On resume, the 4-second unknown interval is excluded.
    assert engine.update("phone", True, 5.0, 1, {}) is None
    assert engine.update("phone", True, 6.9, 1, {}) is None
    assert engine.update("phone", True, 7.0, 1, {}) is not None
