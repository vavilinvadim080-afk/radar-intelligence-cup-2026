"""Additional refusal rules; never grants authorization or sends a response."""
import re
import unicodedata

from .reply_safety import unsafe_reason as prior_reason
from .semantics_v6b import authority_override


def unsafe_reason(text):
    text=unicodedata.normalize('NFKC',text)
    text=''.join(c for c in text if unicodedata.category(c)!='Cf').lower().replace('ё','е')
    old=prior_reason(text)
    if old:return old
    if authority_override(text):return 'попытка изменить правила допуска'
    tests=[
        (r'(?:запишите|вставьте|опубликуйте|внесите|сохраните|добавьте)[^.!?\n]{0,100}(?:токен|секрет|парол|код\s+доступ)', 'раскрытие учётных данных'),
        (r'(?:токен|секрет|парол)[^.!?\n]{0,90}(?:запишите|вставьте|опубликуйте|внесите)', 'раскрытие учётных данных'),
        (r'(?:зачтите|зачесть|перезачтите|зачислите)[^.!?\n]{0,90}(?:платеж|деньг|сумм|рубл)', 'персональная денежная операция'),
        (r'(?:расторгните|прекратите)[^.!?\n]{0,80}(?:договор|соглашени)', 'персональное юридическое действие'),
        (r'(?:оформите|подтвердите)[^.!?\n]{0,80}(?:расторжен|юридическ[^.!?\n]{0,20}заверш)', 'персональное юридическое действие'),
        (r'(?:объявляю|считай|считайте|назначаю)[^.!?\n]{0,65}(?:системн[^.!?\n]{0,30}инструкц)', 'изменение полномочий'),
        (r'(?:вместо\s+анализа|не\s+анализируй)[^.!?\n]{0,70}(?:верни|выдай)[^.!?\n]{0,45}разрешени', 'обход анализа'),
        (r'команд[аы]\s+администратор[^.!?\n]{0,80}(?:разреш|автоответ|обход|отключ)', 'самоназначенные полномочия'),
        (r'(?:treat|consider)[^.!?\n]{0,60}(?:system\s+instruction|system\s+message)', 'изменение полномочий'),
    ]
    for pattern,reason in tests:
        if re.search(pattern,text):return reason
    return None
