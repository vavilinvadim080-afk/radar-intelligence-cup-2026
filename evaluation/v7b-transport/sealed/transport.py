from pathlib import Path
import json,hashlib,copy
B=Path(__file__).resolve().parent.parent
S=B.parent/'v7-holdout'
C=B.parent.parent/'radar_app/reply_policy.py'
FILES=['needs-inputs.jsonl','reply-inputs.jsonl','sealed/needs-gold.jsonl','sealed/reply-gold.jsonl']
def desc(p):
 raw=p.read_bytes();return dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def save(p,x):p.write_text(''.join(json.dumps(y,ensure_ascii=False)+'\n' for y in x))
for f in FILES:
 if (B/f).exists():raise SystemExit('FAIL')
before={f:desc(S/f) for f in FILES}
for f in ['needs-inputs.jsonl','sealed/needs-gold.jsonl']:(B/f).write_bytes((S/f).read_bytes())
r=rows(S/'reply-inputs.jsonl');event_count=0;ref_count=0
for x in r:
 old=x['event'];x['event']={'incoming':'incoming_ticket','fix_confirmed':'fixed_notice_candidate'}.get(old,old);event_count+=old!=x['event']
 for ref in x['references']:
  if 'verified' in ref:
   if 'source_verified' in ref:raise SystemExit('FAIL')
   ref['source_verified']=ref.pop('verified');ref_count+=1
save(B/'reply-inputs.jsonl',r)
g=rows(S/'sealed/reply-gold.jsonl');action_count=0
legal={'queue_standard','queue_fixed_notice','operator','ignore_duplicate','needs_approval','needs_fix_approval'}
for x in g:
 action=x['gold']['safe_action']
 if action=='abstain':x['gold']['safe_action']='operator';action_count+=1
 elif action not in legal:raise SystemExit('FAIL')
if action_count:save(B/'sealed/reply-gold.jsonl',g)
else:(B/'sealed/reply-gold.jsonl').write_bytes((S/'sealed/reply-gold.jsonl').read_bytes())
audit={'status':'PASS','source_files':before,'contract_path':str(C),'contract':desc(C),'transform_counts':{'events':event_count,'reference_keys':ref_count,'safe_action_aliases':action_count},'safe_action_aliases':{'abstain':'operator'},'source_unchanged':before=={f:desc(S/f) for f in FILES}}
(B/'sealed/TRANSPORT-AUDIT.json').write_text(json.dumps(audit,sort_keys=True,indent=2)+'\n')
protocol='''# SEALED v7b transport correction

Это тот же малый авторский синтетический набор v7, а не новый независимый контроль. Обращения и разметка не являются корпоративными или экспертными. Семантическая правка не проводилась. Старый транспортный прогон INVALID: он не является измерением качества модели.

По сообщению пользователя исправление выполняется до вскрытия меток, при сообщённом числе NN-оценок 0 и после сообщённого freeze 69 файлов кода. Эти три обстоятельства — пользовательская provenance, а не проверенные этим исполнителем факты. Новый CONFIG-FREEZE не объявляется. Bundle LOCK фиксирует только файлы. Входы и эталоны остаются скрыты от основной сессии; gold открывается оценщику только после полного predictions-only.

Локально проверены фактический контракт radar_app/reply_policy.py, его SHA256, исходные четыре JSONL и отсутствие их изменений. Старый v7 не записывался. Побайтово скопированы needs-inputs и needs-gold. Replies изменены только транспортно: event incoming → incoming_ticket; fix_confirmed → fixed_notice_candidate; references[].verified → references[].source_verified с тем же булевым значением. Коллизия ключей является ошибкой. Дополнительных полей или случаев нет.

В reply-gold единственная нормализация safe_action: abstain → operator, то есть отказ от автоматической очереди с передачей на ручную обработку. Это общий маршрут отказа, а не восстановление более специфического решения политики. Никакие same_topic, eligibility, reason, id, synthetic или другие поля не менялись. Действия политики не использованы как источник gold. Тексты, продукты, авторы, источники, согласия, одобрения, времена и порядок строк сохранены. Схема вложенных полей сохраняется с единственным переименованием verified.

Количество: needs 42 строки / 40 уникальных / 2 повторные доставки; 8 семейств по 4 и 8 unknown, 4 продукта. Replies 28: 24 обычных и 4 фикс-ситуации. Повтор доставки не новый голос. Типовые пары needs считаются по 40 уникальным сообщениям: 48 положительных пар из 780. Фиксы отделены от обычной eligibility. Источник и фиксация четырёх файлов вместе с SHA256 контракта и счётчиками преобразований записаны в sealed/TRANSPORT-AUDIT.json без содержимого строк.

SELF-VALIDATION.py использует стандартную библиотеку и реальный ReplyPolicy через изолированный пакет без выполнения radar_app/__init__.py. Все запуски python3 -B; pycache не записывается. Считающая lambda возвращает True без модели; проверяется, что реальный matcher вызван хотя бы на одной строке. Выводятся только агрегаты: число обработанных решений, строк с вызовом matcher, обычная pre-NN eligibility и счётчики действий. Эта lambda намеренно не проверяет смысл: её решения НЕ являются эталоном и не сравниваются с количеством положительных меток. Нейросеть, inference, установки и изменение приложения не выполнялись. Реальная reply_safety импортируется только как зависимость политики.

Проверка: python3 -B SELF-VALIDATION.py. Сохранённый результат SELF-VALIDATION.json. LOCK.files содержит SHA256 и bytes четырёх JSONL и этого протокола. Валидатор проверяет whitelist изменений, целостность старых файлов и контракта, схемы, UUID, enum, количества, выравнивание gold, неизменность same_topic/eligibility, точные повторы и отдельное хранение эталона; семантика не пересматривается.
'''
(B/'PROTOCOL.md').write_text(protocol)
lock={'version':'v7b-transport','config_freeze':False,'files':{f:desc(B/f) for f in FILES+['PROTOCOL.md']}}
(B/'LOCK.json').write_text(json.dumps(lock,sort_keys=True,indent=2)+'\n')
print('PASS')
