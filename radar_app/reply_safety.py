"""Individual requests requiring a human even when a known issue is mentioned."""
import re

INJECTION=re.compile(r'игнорируй.{0,60}(?:инструк|правил)|ignore.{0,60}(?:instruction|rule)|system\s*prompt',re.I|re.S)
CREDENTIALS=re.compile(r'(?:пришл|отправ|покаж|сообщ|сброс|выда|измен|поменя|сме[нн])[а-я ]{0,30}(?:парол|секрет|код(?:а|ы)?\b|доступ)|(?:парол|секрет).{0,25}(?:пришл|отправ|покаж|сообщ|сброс)|<(?:PASSWORD|SECRET|TOKEN)[_\d]*>',re.I)
MONEY=re.compile(r'(?:верн|перевед|возмест|компенсир|спиш|зачислите|оформ.{0,15}возврат).{0,65}(?:ден[еьг]|рубл|счет|карт|оплат|платеж|сумм)|(?:возврат|перевод) (?:денег|платежа)|(?:ден[еьг]|рубл).{0,45}(?:верн|перевед|компенсир)',re.I|re.S)


def unsafe_reason(text):
    s=text.lower().replace('ё','е')
    if INJECTION.search(s):return 'маркер инъекции'
    if CREDENTIALS.search(s):return 'индивидуальное действие с учётными данными'
    if MONEY.search(s):return 'индивидуальное денежное действие'
    return None
