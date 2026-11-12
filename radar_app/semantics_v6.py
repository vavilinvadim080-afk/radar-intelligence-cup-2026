"""Conservative action polarity and independent authority-override refusal."""
import re
import unicodedata
from .semantic import normalize
from .selection_v5c import functionality_state

_POSITIVE = re.compile(r'^(?:включ|добав|присыл|отправ|получ|приход|достав|разреш|активир|возобнов|восстанов|вернит|вернуть)')
_NEGATIVE = re.compile(r'^(?:выключ|отключ|убр|убер|удал|исключ|прекрат|прекращ|переста|отмен|запрет|отпис)')
_VOLITION = re.compile(r'^(?:хоч|хотел|прошу|просим|просил|нужн|надо|пожалуйста|позвол|желал|желан)')
_STOP = re.compile(r'^(?:прекрат|прекращ|переста)')
_SYMPTOM = re.compile(r'не\s+(?:получа[юе][а-я]*|приход[а-я]*|доставля[а-я]*|поступа[а-я]*)')


def requested_direction(text):
    tokens = re.findall(r'[а-яa-z]+', normalize(text))
    explicit, inferred = [], []
    for i, token in enumerate(tokens):
        sign = -1 if _NEGATIVE.match(token) else 1 if _POSITIVE.match(token) else 0
        if not sign:
            continue
        before = tokens[max(0, i - 7):i]
        volition = any(_VOLITION.match(t) for t in before)
        past_stop = bool(_STOP.match(token) and token.endswith(('ли', 'ла', 'ло', 'л')))
        if past_stop and not volition:
            continue
        if not volition and re.search(r'(?:ен|ена|ены|ено|енный|енная)$', token):
            continue  # The current enabled/disabled state is not a request.
        previous_stops = [t for t in before[-3:] if _STOP.match(t)]
        if sign > 0 and previous_stops:
            if not volition and any(t.endswith(('ли', 'ла', 'ло', 'л')) for t in previous_stops):
                inferred.append(1)
            continue  # "Stop sending" is a single action, not two requests.
        negations = before.count('не')
        inability = any(t in {'могу', 'можем', 'невозможно'} for t in before)
        symptom = bool(_SYMPTOM.search(' '.join(tokens[max(0, i - 1):i + 1])))
        if not volition and (symptom or inability):
            inferred.append(sign)
        else:
            explicit.append(sign * (-1 if negations % 2 else 1))
    if not explicit and _SYMPTOM.search(' '.join(tokens)):
        inferred.append(1)
    signs = set(explicit or inferred)
    return 'enable' if signs == {1} else 'disable' if signs == {-1} else None


def compatible_direction(a, b):
    left, right = requested_direction(a), requested_direction(b)
    if left is not None and right is not None and left != right:
        return False
    left, right = functionality_state(a), functionality_state(b)
    return left is None or right is None or left == right


def authority_override(text):
    text = unicodedata.normalize('NFKC', text)
    text = normalize(''.join(c for c in text if unicodedata.category(c) != 'Cf'))
    patterns = [
        r'(?:назнач|присва)[^.!?]{0,70}(?:текст|сообщени|инструкц)[^.!?]{0,70}(?:высш|максимальн)[^.!?]{0,25}приоритет',
        r'(?:игнор|забуд|отброс)[^.!?]{0,55}(?:предыдущ|прежн|системн)[^.!?]{0,35}(?:инструкц|правил)',
        r'(?:регламент|правил)[^.!?]{0,35}(?:допуск|соглас|одобр|безопасност)[^.!?]{0,35}(?:отмен|недейств)',
        r'(?:считай|считайте|полагай)[^.!?]{0,45}(?:разрешени|согласи|одобр)[^.!?]{0,40}(?:получ|дано|есть)',
        r'не\s+проверя[а-я]*[^.!?]{0,35}(?:согласи|одобр|разрешени|допуск)',
        r'(?:override|ignore|disregard)[^.!?]{0,60}(?:previous|system|prior)[^.!?]{0,35}(?:instruction|rule)',
        r'(?:skip|bypass)[^.!?]{0,35}(?:approval|permission|consent|safety)[^.!?]{0,25}(?:check|validation)',
    ]
    return any(re.search(pattern, text) for pattern in patterns)
