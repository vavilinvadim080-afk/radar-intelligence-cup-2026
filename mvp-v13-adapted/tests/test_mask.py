from radar.gen import generate
from radar.mask import mask_text, mask_tickets


def test_basic_pii_replaced_and_mapping_returned():
    text = "Мой номер +7 (912) 345-67-89, почта ivan@mail.ru, авто А123ВС55, карта 4276 1234 5678 9012. Иванов Сергей Петрович"
    masked, mapping = mask_text(text)
    for secret in ["345-67-89", "ivan@mail.ru", "А123ВС55", "5678 9012", "Иванов", "Сергей", "Петрович"]:
        assert secret not in masked
    for token in ["<PHONE_1>", "<EMAIL_1>", "<CAR_PLATE_1>", "<CARD_1>", "<PERSON_1>"]:
        assert token in masked
    assert mapping["<PHONE_1>"] == "+7 (912) 345-67-89"      # оригиналы лежат отдельно
    assert all(v not in masked for v in mapping.values())     # а в тексте модели их нет


def test_same_value_gets_same_token_and_plain_text_unchanged():
    masked, _ = mask_text("Звоните 89123456789. Повторяю: 89123456789")
    assert masked.count("<PHONE_1>") == 2 and "<PHONE_2>" not in masked
    plain = "Приложение не работает, оплата не прошла"
    assert mask_text(plain)[0] == plain


def test_completeness_on_generated_pii():
    """Генератор знает, что вставил. Простые форматы должны маскироваться все, трудные (склонения,
    латиница, 'собака') могут утекать, но не больше 15% при низком шуме."""
    ds = generate(5, "low", 500)
    texts, _ = mask_tickets(ds.tickets)
    inserted = leaked = 0
    for tk, masked in zip(ds.tickets, texts):
        for kind, value in ds.labels[tk.id]["pii"]:
            inserted += 1
            digits = "".join(c for c in value if c.isdigit())
            if kind in ("phone", "card"):
                bad = digits[:5] in "".join(c for c in masked if c.isdigit())
            elif kind == "email":
                bad = value.split("@")[0] in masked
            else:
                bad = any(t in masked for t in value.replace(".", " ").split() if len(t) > 3 and t[0].isupper())
            leaked += bad
    assert inserted > 200
    assert leaked / inserted < 0.15, f"утекло {leaked} из {inserted}"
