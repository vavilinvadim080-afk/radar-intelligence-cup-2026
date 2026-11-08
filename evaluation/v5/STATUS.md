# Независимый контроль v5

- [x] Прочитаны спецификация и план 2026-11-04-quality-teaching-v5; зоны записи зафиксированы.
- [x] До gold-lock новый semantic matcher и старые demo_analyze rules не читались. Рабочий код не меняется.
- [x] Синтетические tune/control обращения: 120+120, 24+24 независимых определённых семейства; product/kind в gold topic ID; неоднозначное остаётся unknown.
- [x] Reply-pairs 80+80: same_topic и eligibility разделены; отдельные safety/approval/fix/duplicate пути.
- [x] Контрольные inputs без gold/expected; ответы отдельно; SHA-256 шести файлов, счёт, протокол и дата в CONTROL-LOCK.json. Данные read-only; временный источник с контрольными словарями удалён, канонические JSONL сохранены.
- [x] validate.py: pass до и после lock. Проверены схема, ID, разделение целей/текстов split, повторы, permissions, отсутствие gold в inputs, хэши. Никаких model runs.
- [x] Handoff только tune/пути/счёт; контрольные ответы/словари не раскрываются. Дальше Task 5.
- [ ] После lock — полный учебник Task 5, с локальными markdown-источниками; финальные snippets и измерения после готовности data/quality-v5.json и load-effect-v5.json.

Промо-case-v1 и O не меняются. Root README, Мозг, backend и presentation не редактируются. Тяжёлых модельных задач здесь пока нет.
