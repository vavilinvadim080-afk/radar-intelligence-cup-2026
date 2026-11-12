"""Experimental matching inside the complete, unchanged authorization policy."""
import hashlib
import json
import math

from .reply_policy import ReplyPolicy, _reject_gold, FIELDS
from .reply_safety_v7 import unsafe_reason
from .selection_v5b import RISK_HYPOTHESES


def pair_key(a, b):
    return hashlib.sha256(json.dumps([a, b], ensure_ascii=False).encode()).hexdigest()


class ReplyMatcherV7:
    def __init__(self, scores, threshold=.95, risks=None, blocked=None):
        self.scores = scores
        self.threshold = threshold
        self.risks = risks or {}
        self.blocked = blocked or {}

    def match(self, payload):
        refs, text = payload['references'], payload['query']['text']
        if any(ref.get('kind') != payload['kind'] for ref in refs):
            return False
        # Either ordering can recognize a specific symptom inside a broad example.
        return any(max(self.scores[pair_key(text, ref['text'])],
                       self.scores[pair_key(ref['text'], text)]) >= self.threshold for ref in refs)

    def decide(self, payload):
        _reject_gold(payload)
        if set(payload)-FIELDS:
            raise ValueError('Unknown policy fields')
        query = payload.get('query', {})
        if query.get('id') in self.blocked and query.get('id') not in payload.get('seen_message_ids',[]):
            return dict(action='operator',reason=self.blocked[query['id']],same_topic=None,eligible=False)
        if (query.get('id') not in payload.get('seen_message_ids', [])
                and unsafe_reason(query.get('text', ''))):
            return dict(action='operator', reason=unsafe_reason(query.get('text','')), same_topic=None, eligible=False)
        if query.get('id') in payload.get('seen_message_ids', []):
            return ReplyPolicy(self.match).decide(payload)
        risks=self.risks.get(hashlib.sha256(query.get('text','').encode()).hexdigest(), {})
        if (set(risks)!=set(RISK_HYPOTHESES) or any(isinstance(v,bool) or not isinstance(v,(int,float))
                or not math.isfinite(v) or not 0<=v<=1 for v in risks.values())):
            return dict(action='operator',reason='проверка риска недоступна',same_topic=None,eligible=False)
        if any(v>=.6 for v in risks.values()):
            return dict(action='operator',reason='индивидуальное действие или инструкция требует человека',same_topic=None,eligible=False)
        return ReplyPolicy(self.match).decide(payload)


def requests_for(rows, risks=None, blocked=None):
    """Include pairs only when the full policy actually reaches its matcher."""
    requests = {}
    for row in rows:
        called = []
        probe = ReplyMatcherV7({}, risks=risks, blocked=blocked)
        probe.match = lambda payload: called.append(True) or True
        probe.decide(row)
        if not called:
            continue
        for ref in row['references']:
            a, b = row['query']['text'], ref['text']
            for left, right in [(a, b), (b, a)]:
                requests[pair_key(left, right)] = [left, right]
    return requests
