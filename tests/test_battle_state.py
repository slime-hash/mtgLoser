from mcoc_vision.battle_state import BattleState


def test_pause_unknown_and_next_battle():
    state = BattleState(confirm_seconds=.3)
    assert not state.observe('combat', .95, 0.)
    first = state.observe('combat', .95, .5)[0]['fight_id']
    assert not state.observe('unknown', .1, 1.)
    assert state.fight_id == first
    state.observe('paused', .95, 2.)
    assert state.observe('paused', .95, 2.5)[0]['event'] == 'pause'
    state.observe('combat', .95, 3.)
    assert state.observe('combat', .95, 3.5)[0]['fight_id'] == first
    state.observe('result', .95, 4.)
    assert state.observe('result', .95, 4.5)[0]['partial'] is False
    state.observe('combat', .95, 5.)
    assert state.observe('combat', .95, 5.5)[0]['fight_id'] != first


def test_low_confidence_breaks_transition_confirmation():
    state = BattleState()
    state.observe('combat', .99, 0.)
    state.observe('combat', .2, .5)
    assert not state.observe('combat', .99, 1.)
    assert state.observe('combat', .99, 1.5)[0]['event'] == 'start'
    assert state.stop()[0]['partial']
