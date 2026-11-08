from radar_app.core import Radar


class LocalAnalyzer:
    mode = 'local-semantic-prototype'
    minimum_confidence = 0
    supports_examples = True

    def __init__(self): self.catalogs = []

    def __call__(self, text, catalog):
        self.catalogs.append(catalog)
        return dict(topic_key='new-topic', kind='wish', label='Новая потребность', quote=text, confidence=.43)


def test_local_mode_is_explicit_and_dynamic_catalog_contains_only_product_scoped_masked_evidence(tmp_path):
    a = LocalAnalyzer()
    r = Radar(tmp_path/'db', secret='semantic-test-key-long', now='2026-11-01T12:00:00Z', analyzer=a, ner=False)
    def ticket(product, ident, text):
        return dict(product=product, id=ident, author=ident, created_at='2026-10-30T12:00:00Z', text=text)
    try:
        first = r.ingest(ticket('one', 't1', 'Хочу сохранять покупки, почта user@example.test'))
        assert first['status'] == 'accepted'
        assert first['mode'] == 'local-semantic-prototype'
        r.ingest(ticket('one', 't2', 'Дайте историю файлом'))
        examples = a.catalogs[-1][0]['examples']
        assert examples and 'user@example.test' not in str(examples)
        r.ingest(ticket('two', 't3', 'В другом продукте тоже хочу историю'))
        assert a.catalogs[-1] == []
    finally: r.close()


def test_internal_gateway_confidence_gate_is_unchanged(tmp_path):
    a = LocalAnalyzer()
    r = Radar(tmp_path/'db', secret='semantic-test-key-long', now='2026-11-01T12:00:00Z',
              analyzer=lambda text,catalog: a(text,catalog), ner=False)
    try:
        out = r.ingest(dict(product='one', id='t1', author='u', created_at='2026-10-30T12:00:00Z', text='Хочу историю'))
        assert out['status'] == 'unknown' and out['mode'] == 'internal-model'
    finally: r.close()
