from radar.config import Config
from radar.gen import generate
from radar.pipeline import card_input, group_incidents, run_full


def test_pipeline_smoke_and_masked_examples():
    ds = generate(7, "low", 400)
    st = run_full(ds.tickets, ds.releases, Config(use_hdbscan=False, min_users=5))
    ids = {t.id for t in ds.tickets}
    assert st.bursts, "на 28 днях с регрессиями должен быть хотя бы один всплеск"
    for b in st.bursts:
        assert b.examples and all(e["id"] in ids for e in b.examples)
        assert b.weekly_cost_rub >= 0 and b.total_unique_users >= 1
        card = card_input(st, b)
        assert set(card.allowed_ids) <= ids
    # в текстах для агента нет ни одного оригинала простых ПДн
    pii_values = [v for t in ds.tickets for k, v in ds.labels[t.id]["pii"] if k == "email" and "@" in v]
    blob = " ".join(e["text"] for b in st.bursts for e in b.examples)
    assert all(v not in blob for v in pii_values)
    assert sum(len(g) for g in group_incidents(st.bursts)) == len(st.bursts)
