from risk_window.runtime import HysteresisState, update_hysteresis


def test_hysteresis_enters_after_three_hits_and_exits_after_five_hits():
    state = HysteresisState()
    events = []
    for score in (0.71, 0.72, 0.75):
        events.append(update_hysteresis(state, score, True, 0.70, 3, 0.45, 5))
    assert state.in_window is True
    assert events[-1] == "enter_window"

    for score in (0.44, 0.43, 0.42, 0.40):
        assert update_hysteresis(state, score, True, 0.70, 3, 0.45, 5) is None
        assert state.in_window is True
    assert update_hysteresis(state, 0.39, True, 0.70, 3, 0.45, 5) == "exit_window"
    assert state.in_window is False
