"""Temporal battle boundaries driven by an independent visual screen classifier.

No detection counts are used: losing a fighter must not terminate a battle.
The visual classifier is a separate, not-yet-integrated component.
"""
from uuid import uuid4


class BattleState:
    def __init__(self, confirm_seconds=.4, min_confidence=.8):
        self.confirm_seconds = confirm_seconds
        self.min_confidence = min_confidence
        self.fight_id = None
        self.pending = None
        self.pending_at = None
        self.last_at = None
        self.confirmed = 'unknown'

    def observe(self, state, confidence, timestamp):
        if state not in {'unknown', 'menu', 'loading', 'intro', 'combat', 'paused', 'result'}:
            raise ValueError('Unknown screen state')
        if self.last_at is not None and timestamp <= self.last_at:
            raise ValueError('Screen timestamps must increase')
        self.last_at = timestamp
        if confidence < self.min_confidence or state == 'unknown':
            self.pending = None
            return []
        if state != self.pending:
            self.pending, self.pending_at = state, timestamp
        if timestamp - self.pending_at < self.confirm_seconds or state == self.confirmed:
            return []
        self.confirmed = state
        if state == 'combat':
            if self.fight_id is None:
                self.fight_id = uuid4().hex
                return [dict(event='start', fight_id=self.fight_id)]
            return [dict(event='resume', fight_id=self.fight_id)]
        if self.fight_id is not None:
            if state == 'paused':
                return [dict(event='pause', fight_id=self.fight_id)]
            if state in {'result', 'loading', 'menu', 'intro'}:
                event = dict(event='end', fight_id=self.fight_id,
                             partial=state != 'result', reason=state)
                self.fight_id = None
                return [event]
        return []

    def stop(self):
        events = []
        if self.fight_id is not None:
            events.append(dict(event='end', fight_id=self.fight_id,
                               partial=True, reason='session_stopped'))
        self.fight_id = None
        self.confirmed = 'unknown'
        self.pending = None
        return events
