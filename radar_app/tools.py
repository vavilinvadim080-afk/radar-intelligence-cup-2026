"""Small deterministic tool boundary. No arbitrary network, SQL or user messaging."""
import hashlib
import json

TOOL_GUIDES={
    'list_topics':'Без аргументов. Темы текущего продукта за 28 дней с уникальными пользователями и источниками; чтение.',
    'get_evidence':'Аргумент topic. Дословные замаскированные цитаты текущей карточки; чтение.',
    'search_tracker':'Аргумент topic. Существующая задача с совпадающими продуктом, темой и типом; чтение конфигурируемого адаптера.',
    'queue_draft':'Аргумент topic. Только после решения accepted; идемпотентный черновик в локальной очереди. Никакой отправки пользователю.',
    'queue_digest':'Без аргументов. Идемпотентная сводка доступных карточек в локальной очереди. Доставщик компании подключается отдельно.',
}

class ProductTools:
    def __init__(self,radar,product):
        self.radar=radar;self.product=product
        with radar.lock,radar.db:
            radar.db.execute('CREATE TABLE IF NOT EXISTS outbox(id TEXT PRIMARY KEY,product TEXT,kind TEXT,payload TEXT,status TEXT,created TEXT)')

    def call(self,name,args):
        if name not in TOOL_GUIDES or not isinstance(args,dict):raise ValueError('Unsupported tool')
        required={'topic'} if name in {'get_evidence','search_tracker','queue_draft'} else set()
        if set(args)!=required:raise ValueError('Unexpected tool arguments')
        with self.radar.lock:
            cards=self.radar.digest(self.product)
            if name=='list_topics':return cards
            if name=='queue_digest':
                return self._queue('digest',{'cards':cards,'period_end':self.radar.now().date().isoformat()})
            c=next((c for c in cards if c['id']==args['topic']),None)
            if c is None:raise ValueError('Unknown topic in product')
            if name=='get_evidence':return c['evidence']
            if name=='search_tracker':return {'existing_task':c['existing_task']}
            if c['existing_task']:return {'existing_task':c['existing_task'],'status':'already_exists'}
            if c['status']!='accepted':raise ValueError('Product decision required')
            return self._queue('draft',c)

    def _queue(self,kind,payload):
        text=json.dumps(payload,ensure_ascii=False,sort_keys=True)
        # Draft id stays stable as evidence changes; digest includes the period and content.
        identity=payload['id'] if kind=='draft' else text
        key=hashlib.sha256((self.product+':'+kind+':'+identity).encode()).hexdigest()[:24]
        with self.radar.db:
            self.radar.db.execute('INSERT OR IGNORE INTO outbox VALUES(?,?,?,?,?,?)',(key,self.product,kind,text,'pending_delivery',self.radar.now().isoformat()))
        return {'id':key,'status':'pending_delivery','product':self.product}
