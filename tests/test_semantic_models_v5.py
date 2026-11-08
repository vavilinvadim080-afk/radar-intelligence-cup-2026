import json
import math
import os
from pathlib import Path

import pytest

from radar_app.semantic_client import SemanticClient


@pytest.mark.skipif(os.environ.get('RADAR_TEST_LOCAL_MODELS') != '1', reason='Explicit offline model check')
def test_cached_multilingual_models_encode_and_rerank_without_network(tmp_path):
    texts = ['Нужна выгрузка истории заказов в файл',
             'Хочу скачать прошлые заказы таблицей',
             'Не приходит смс для входа']
    with SemanticClient(cache_dir=tmp_path / 'cache') as c:
        vectors = c.embed(texts)
        desc = c.describe()
        assert desc['dimension'] == 384 and desc['offline'] is True
        assert all(len(v) == 384 and math.isclose(sum(x*x for x in v), 1, abs_tol=1e-5) for v in vectors)
        dot = lambda a, b: sum(x*y for x, y in zip(a, b))
        assert dot(vectors[0], vectors[1]) > dot(vectors[0], vectors[2])
        scores = c.rerank([(texts[0], texts[1]), (texts[0], texts[2])])
        assert scores[0] > scores[1]
        single = c.rerank([('Нужен отдельный файл заказов', 'Хочу скачать историю покупок')])
        assert len(single)==1 and math.isfinite(single[0])
        result = {'texts': texts, 'metadata': desc, 'first_eight_components': [v[:8] for v in vectors],
                  'cosine': [[dot(a, b) for b in vectors] for a in vectors], 'reranker_logits': scores}
        Path('reports/semantic-models-v5-smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
