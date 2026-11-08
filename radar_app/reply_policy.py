"""Permission and lifecycle gates around matching; only prepares a code template."""
from .reply_safety import unsafe_reason

FIELDS={'event','product','kind','references','query','approved','state','fix_confirmed',
        'human_closed','fixed_notice_approved','seen_message_ids','id'}


def payload_only(row):
    return {k:v for k,v in row.items() if k in FIELDS}


def _reject_gold(value):
    if isinstance(value,dict):
        if any(k in {'gold','expected','same_topic','eligibility','safe_action','topic_id'} for k in value):
            raise ValueError('Evaluation answers cannot enter working policy')
        for v in value.values():_reject_gold(v)
    elif isinstance(value,list):
        for v in value:_reject_gold(v)


class ReplyPolicy:
    def __init__(self,matcher):self.matcher=matcher

    @staticmethod
    def eligibility(p):
        q=p.get('query',{});refs=p.get('references',[])
        if not refs or not all(r.get('source_verified') is True for r in refs):return False,'непроверенные источники'
        if q.get('consent') is not True:return False,'нет согласия'
        risk=unsafe_reason(q.get('text',''))
        if risk:return False,risk
        if q.get('id') in p.get('seen_message_ids',[]):return False,'повторная доставка'
        if p.get('approved') is not True:return False,'нужно одобрение'
        if p.get('fix_confirmed') or p.get('human_closed') or p.get('state')!='open' or p.get('event')!='incoming_ticket':
            return False,'отдельный путь проверки фикса'
        return True,''

    def decide(self,p):
        _reject_gold(p)
        if set(p)-FIELDS:raise ValueError('Unknown policy fields')
        q=p.get('query',{});refs=p.get('references',[])
        eligible,why=self.eligibility(p)
        def out(action,reason,same=None):
            return dict(action=action,reason=reason,same_topic=same,eligible=eligible)
        if q.get('id') in p.get('seen_message_ids',[]):return out('ignore_duplicate','повторная доставка')
        if not refs or not all(r.get('source_verified') is True for r in refs):return out('operator','непроверенные источники')
        if q.get('consent') is not True:return out('operator','нет согласия')
        risk=unsafe_reason(q.get('text',''))
        if risk:return out('operator',risk)
        if q.get('product')!=p.get('product') or any(r.get('product')!=p.get('product') for r in refs):
            return out('operator','другой продукт',False)
        try:same=self.matcher(p)
        except Exception:return out('operator','смысловая проверка недоступна')
        if same is not True:return out('operator','тема не подтверждена',same)
        if p.get('approved') is not True:return out('needs_approval','нужно одобрение',same)
        if (p.get('event')=='fixed_notice_candidate' and p.get('fix_confirmed') is True
                and p.get('human_closed') is True and p.get('fixed_notice_approved') is True):
            return out('queue_fixed_notice','отдельная подтверждённая рассылка',same)
        if p.get('fix_confirmed') or p.get('human_closed') or p.get('event')=='fixed_notice_candidate':
            return out('needs_fix_approval','нужны подтверждения фикса и рассылки',same)
        if p.get('state')!='open':return out('operator','инцидент не открыт',same)
        return out('queue_standard','совпавшая безопасная тема',same)
