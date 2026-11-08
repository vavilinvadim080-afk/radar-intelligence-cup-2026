"""Read saved message history. No analyzer, detector or model is called here.

Needs use their existing SQLite records. Legacy replay lacks complete cluster
membership and author ids: its separate read index contains only saved evidence
ids and already recorded routing. Unlinked messages stay explicitly unlinked.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Literal
from urllib.parse import urlsplit

from fastapi import HTTPException, Query, Request

UNASSIGNED = '__unassigned__'


class InvalidCursor(ValueError):
    pass


class SnapshotExpired(ValueError):
    pass


class UnknownTopic(ValueError):
    pass


def time_key(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        # Replay timestamps are model-calendar time; this is only a sort key.
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec='microseconds')


def bound(value, end=False):
    if not value:
        return ''
    date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if end and len(value) == 10:
        date += timedelta(days=1)
    return time_key(date)


def source_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {'https', 'http'} and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return None


class NeedsHistory:
    kind = 'needs'
    coverage = 'saved_analysis'
    notice = ('История читает сохранённый анализ. Оригиналы и имена авторов в этой базе не сохраняются. '
              'Новые поступления появляются после обновления снимка.')
    rows_sql = '''SELECT rowid AS seq, product,id,created AS created_at,
      CASE WHEN instr(created,'.')=0 THEN replace(created,'+00:00','.000000+00:00') ELSE created END AS time_key,
      author AS author_id,NULL AS author_name,'hmac' AS author_kind,channel,status,text,
      NULL AS original_text,NULL AS source_url,
      CASE WHEN status='accepted' AND json_valid(analysis)
          AND json_extract(analysis,'$.kind') IS NOT NULL AND json_extract(analysis,'$.topic_key') IS NOT NULL
        THEN json_extract(analysis,'$.kind')||':'||json_extract(analysis,'$.topic_key') ELSE '__unassigned__' END AS topic_id,
      CASE WHEN json_valid(analysis) THEN json_extract(analysis,'$.label') END AS topic_title FROM tickets'''

    def __init__(self, radar):
        self._owner = radar
        self.db, self.lock = radar.db, radar.lock
        self.mode, self.clock = radar.mode, radar.now
        self.clock_label = 'модельное время' if getattr(radar, 'fixed_now', None) is not None else 'время снимка'
        self.generation, self.owned = secrets.token_hex(12), False

    @classmethod
    def from_path(cls, path, mode):
        instance = cls.__new__(cls)
        path = Path(path)
        instance.lock = threading.RLock()
        instance.mode, instance.clock = mode, lambda: datetime.now(timezone.utc)
        instance.clock_label = 'время снимка'
        instance.generation, instance.owned = secrets.token_hex(12), True
        if path.is_file():
            instance.db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, check_same_thread=False)
            instance.db.execute('PRAGMA query_only=ON')
        else:
            # Reading an empty context must not create a runtime DB or load NN.
            instance.db = sqlite3.connect(':memory:', check_same_thread=False)
            instance.db.execute('''CREATE TABLE tickets(product,id,author,created,status,text,analysis,channel)''')
        instance.db.row_factory = sqlite3.Row
        return instance

    def sync(self):
        pass

    def member_high(self):
        return 0

    def close(self):
        if self.owned:
            self.db.close()


class LegacyHistory:
    kind = 'incidents'
    coverage = 'confirmed_ids_only'
    notice = ('Принадлежность теме неполная: запись содержит примеры и уже сохранённую маршрутизацию, '
              'а не все ID кластера. Без сохранённой привязки не означает «не относится к теме». '
              'ID авторов в обычном Replay не сохранены. Новые поступления — после обновления снимка.')
    rows_sql = 'SELECT *,NULL AS topic_id,NULL AS topic_title FROM messages'
    clock_label = 'модельное время'

    def __init__(self, radar):
        self.radar, self.lock = radar, radar.lock
        self._rep, self._world = None, None
        self.db = None
        self.sync()

    @property
    def mode(self):
        return self.radar.mode

    def clock(self):
        return self.radar.now

    def _reset(self):
        if self.db is not None:
            self.db.close()
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
          CREATE TABLE messages(seq INTEGER PRIMARY KEY AUTOINCREMENT,product TEXT,id TEXT,created_at TEXT,
            time_key TEXT,author_id TEXT,author_name TEXT,author_kind TEXT,channel TEXT,status TEXT,text TEXT,
            original_text TEXT,source_url TEXT,UNIQUE(product,id));
          CREATE INDEX history_order ON messages(product,time_key DESC,id DESC);
          CREATE TABLE topics(product TEXT,id TEXT,title TEXT,visible_key TEXT,PRIMARY KEY(product,id));
          CREATE TABLE members(seq INTEGER PRIMARY KEY AUTOINCREMENT,product TEXT,id TEXT,topic TEXT,
            observed_key TEXT,origin TEXT,UNIQUE(product,id,topic));
          CREATE INDEX history_members ON members(product,topic,id,seq);
        ''')
        self.generation = secrets.token_hex(12)
        self._indexed = 0
        self._members_indexed = {}
        self._rep, self._world = self.radar.rep, self.radar.world

    def coverage_for(self, product, topic):
        recorded = getattr(self.radar.rep, 'history_members', {})
        cards = [c for c in self.radar.cards.values() if (product is None or c['product'] == product) and (not topic or c['id'] == topic)]
        complete = cards and all(c.get('burst_ids') and all(b in recorded for b in c['burst_ids']) for c in cards)
        if complete and topic != UNASSIGNED:
            return ('saved_stream_membership',
                    'Все сохранённые привязки детектора на момент снимка; поздние слияния не добавлены задним числом. '
                    'Принадлежность к кластеру — результат детектора. Автор — ID из источника, если он сохранён. '
                    'Новые поступления — после обновления снимка.')
        notice = self.notice
        if getattr(self.radar.rep, 'history_authors', {}):
            notice = ('Принадлежность теме неполная для карточек без записи участников потока: доступны сохранённые '
                      'примеры и маршрутизация. Без сохранённой привязки не означает «не относится к теме». '
                      'Автор — ID из источника, если он сохранён. Новые поступления — после обновления снимка.')
        return self.coverage, notice

    def _link(self, product, topic, ids, observed, origin):
        for ident in set(ids):
            if not isinstance(ident, str):
                continue
            self.db.execute('''INSERT OR IGNORE INTO members(product,id,topic,observed_key,origin)
                SELECT product,id,?,?,? FROM messages WHERE product=? AND id=?''',
                (topic, time_key(observed), origin, product, ident))

    def sync(self):
        r = self.radar
        with self.lock:
            if self._rep is not r.rep or self._world is not r.world:
                self._reset()
            for row in r.rep.tickets[self._indexed:]:
                if isinstance(row, dict):
                    ident, created, product = row['id'], row['ts'], row['product']
                    author, name = row.get('author_id'), row.get('author_name')
                    channel, text = row.get('channel', ''), row.get('masked', row.get('text', ''))
                    original, url = row.get('original'), source_url(row.get('source_url'))
                else:
                    ident, created, product, _, _, channel, original, text, _ = row
                    author = getattr(r.rep, 'history_authors', {}).get(product, {}).get(ident)
                    name = url = None
                shown_time = created.isoformat() if isinstance(created, datetime) else created
                self.db.execute('''INSERT OR IGNORE INTO messages(product,id,created_at,time_key,author_id,
                    author_name,author_kind,channel,status,text,original_text,source_url) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (product, ident, shown_time, time_key(created), author, name,
                     'source' if author else 'unavailable', channel, 'saved', text, original, url))
            self._indexed = len(r.rep.tickets)
            burst_cards = {}
            for card in r.cards.values():
                product, ident = card['product'], card['id']
                agent_card = (card.get('agent') or {}).get('card') or {}
                self.db.execute('INSERT OR REPLACE INTO topics VALUES(?,?,?,?)',
                    (product, ident, agent_card.get('title') or ident, time_key(card['created_sim'])))
                for bid in card.get('burst_ids', [ident]):
                    burst_cards[bid] = card
            for bid, card in burst_cards.items():
                records = getattr(r.rep, 'history_members', {}).get(bid, [])
                position = self._members_indexed.get(bid, 0)
                while position < len(records):
                    record = records[position]
                    if time_key(record['observed_at']) > time_key(r.now):
                        break
                    if record['product'] == card['product']:
                        self._link(card['product'], card['id'], [record['id']], record['observed_at'], 'saved_stream_membership')
                    position += 1
                self._members_indexed[bid] = position
            for plan in r.plan:
                card = r.cards.get(plan['card_id'])
                if card:
                    ci = plan['ci']
                    self._link(card['product'], card['id'], getattr(ci, 'allowed_ids', []), plan['ts'], 'saved_card_ids')
            for event in r.rep.events:
                if time_key(event['ts']) > time_key(r.now):
                    continue
                burst = event['burst']; card = burst_cards.get(burst.id)
                if not card or burst.product != card['product']:
                    continue
                ids = [e['id'] for e in getattr(burst, 'examples', []) if isinstance(e, dict) and 'id' in e]
                if event.get('ci'):
                    ids.extend(getattr(event['ci'], 'allowed_ids', []))
                self._link(card['product'], card['id'], ids, event['ts'], 'saved_evidence')
            ar = r.autoreply
            for ident, state in ar.incidents.items():
                card = r.cards.get(ident)
                if not card or state.get('product', card['product']) != card['product']:
                    continue
                ids = list(state.get('writers', []))
                ids.extend(x['ticket_id'] for x in ar.queue + ar.operator if x.get('incident') == ident)
                self._link(card['product'], ident, ids, r.now, 'saved_routing')
            self.db.commit()

    def member_high(self):
        return self.db.execute('SELECT COALESCE(MAX(seq),0) FROM members').fetchone()[0]

    def close(self):
        self.db.close()


class HistoryReader:
    def __init__(self, secret=None):
        self.secret = secret or secrets.token_bytes(32)

    def _encode(self, value):
        body = base64.urlsafe_b64encode(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).decode().rstrip('=')
        return body+'.'+hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()

    def _decode(self, token):
        try:
            if not isinstance(token, str) or len(token) > 8192:
                raise ValueError
            body, signature = token.rsplit('.', 1)
            expected = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            value = json.loads(base64.urlsafe_b64decode(body+'='*(-len(body) % 4)))
            if value['v'] != 1:
                raise ValueError
            return value
        except (ValueError, KeyError, TypeError):
            raise InvalidCursor('Курсор недействителен. Обновите историю.') from None

    @staticmethod
    def _topic(source, topic, snapshot, alias='r'):
        if not topic:
            return '1=1', []
        if source.kind == 'needs':
            return f'{alias}.topic_id=?', [topic]
        clause = (f'EXISTS(SELECT 1 FROM members m WHERE m.product={alias}.product AND m.id={alias}.id '
                  'AND m.seq<=? AND m.observed_key<=?')
        values = [snapshot['members'], snapshot['at']]
        if topic == UNASSIGNED:
            return 'NOT '+clause+')', values
        return clause+' AND m.topic=?)', values+[topic]

    @staticmethod
    def _summary(db, sql, params):
        row = db.execute(sql, params).fetchone()
        total, known, missing = row[0], row[1], row[2] or 0
        return dict(total=total, unique_authors=None if missing else known,
                    known_authors=known, messages_without_author=missing)

    def page(self, source, *, product, topic='', channel='', start='', end='', limit=50, cursor=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Размер страницы: от 1 до 100')
        start_key, end_key = bound(start), bound(end, end=True)
        if start_key and end_key and start_key >= end_key:
            raise ValueError('Начало периода должно быть раньше конца')
        with source.lock:
            source.sync()
            scope = dict(mode=source.kind, source_mode=source.mode, product=product, topic=topic or '',
                         channel=channel or '', start=start_key, end=end_key, limit=limit)
            if cursor:
                token = self._decode(cursor)
                if token.get('scope') != scope:
                    raise InvalidCursor('Курсор относится к другим фильтрам или режиму.')
                snapshot = token['snapshot']
                old_count = source.db.execute('WITH records AS ('+source.rows_sql+') SELECT COUNT(*) FROM records WHERE product=? AND seq<=?',
                                              (product,snapshot['seq'])).fetchone()[0]
                if snapshot['generation'] != source.generation or snapshot.get('record_count') != old_count:
                    raise SnapshotExpired('Источник изменился или был сброшен. Обновите историю.')
            else:
                high,count = source.db.execute('WITH records AS ('+source.rows_sql+') SELECT COALESCE(MAX(seq),0),COUNT(*) FROM records WHERE product=?',(product,)).fetchone()
                now=source.clock()
                snapshot = dict(seq=high,record_count=count,members=source.member_high(),at=time_key(now),
                                shown_at=now.isoformat(),clock=source.clock_label,generation=source.generation)
                token = dict(v=1, scope=scope, snapshot=snapshot, before=None)
            clauses = ['product=?', 'seq<=?', 'time_key<=?']
            params = [product, snapshot['seq'], snapshot['at']]
            for clause, value in [('time_key>=?', start_key), ('time_key<?', end_key), ('channel=?', channel)]:
                if value:
                    clauses.append(clause); params.append(value)
            base = 'WITH records AS ('+source.rows_sql+'), scoped AS (SELECT * FROM records WHERE '+' AND '.join(clauses)+') '
            topic_sql, topic_values = self._topic(source, topic, snapshot)
            if topic and topic != UNASSIGNED:
                if source.kind == 'needs':
                    exists = source.db.execute('WITH records AS ('+source.rows_sql+') SELECT 1 FROM records WHERE product=? AND topic_id=? AND time_key<=? LIMIT 1',
                                               (product, topic, snapshot['at'])).fetchone()
                else:
                    exists = source.db.execute('SELECT 1 FROM topics WHERE product=? AND id=? AND visible_key<=?',
                                               (product, topic, snapshot['at'])).fetchone()
                if not exists:
                    raise UnknownTopic('Тема не найдена в доступной истории.')
            summary = self._summary(source.db, base+'SELECT COUNT(*),COUNT(DISTINCT author_id),SUM(author_id IS NULL) FROM scoped r WHERE '+topic_sql, params+topic_values)
            query, values = base+'SELECT r.* FROM scoped r WHERE '+topic_sql, params+topic_values
            if token.get('before'):
                at, ident = token['before']
                query += ' AND (r.time_key<? OR (r.time_key=? AND r.id<?))'; values += [at, at, ident]
            rows = source.db.execute(query+' ORDER BY r.time_key DESC,r.id DESC LIMIT ?', values+[limit+1]).fetchall()
            has_next = len(rows) > limit; rows = rows[:limit]
            next_cursor = None
            if has_next:
                next_cursor = self._encode({**token, 'before': [rows[-1]['time_key'], rows[-1]['id']]})
            if source.kind == 'needs':
                groups = source.db.execute(base+'''SELECT topic_id AS id,MAX(topic_title) AS title,
                    COUNT(*) AS total,COUNT(DISTINCT author_id) AS authors,SUM(author_id IS NULL) AS missing
                    FROM scoped GROUP BY topic_id ORDER BY total DESC,topic_id LIMIT 201''', params).fetchall()
            else:
                groups = source.db.execute(base+'''SELECT t.id,t.title,COUNT(DISTINCT r.id) AS total,
                    COUNT(DISTINCT r.author_id) AS authors,SUM(r.id IS NOT NULL AND r.author_id IS NULL) AS missing
                    FROM topics t LEFT JOIN members m ON m.product=t.product AND m.topic=t.id AND m.seq<=? AND m.observed_key<=?
                    LEFT JOIN scoped r ON r.product=m.product AND r.id=m.id
                    WHERE t.product=? AND t.visible_key<=? GROUP BY t.id ORDER BY total DESC,t.id LIMIT 201''',
                    params+[snapshot['members'], snapshot['at'], product, snapshot['at']]).fetchall()
                unlinked_sql, unlinked_params = self._topic(source, UNASSIGNED, snapshot)
                unlinked = self._summary(source.db, base+'SELECT COUNT(*),COUNT(DISTINCT author_id),SUM(author_id IS NULL) FROM scoped r WHERE '+unlinked_sql, params+unlinked_params)
                if unlinked['total']:
                    groups = list(groups)+[dict(id=UNASSIGNED, title=None, total=unlinked['total'], authors=unlinked['known_authors'], missing=unlinked['messages_without_author'])]
            topics = [dict(id=g['id'], title=g['title'] or ('Без сохранённой привязки' if g['id'] == UNASSIGNED else g['id']),
                           total=g['total'], unique_authors=None if g['missing'] else g['authors']) for g in groups[:200]]
            channels = [r[0] for r in source.db.execute('WITH records AS ('+source.rows_sql+') SELECT DISTINCT channel FROM records WHERE product=? AND seq<=? AND time_key<=? ORDER BY channel LIMIT 32',
                                                      (product, snapshot['seq'], snapshot['at']))]
            items = [dict(id=r['id'], product=r['product'], created_at=r['created_at'], channel=r['channel'], status=r['status'],
                text=r['text'] or '', original_text=r['original_text'], source_url=source_url(r['source_url']),
                author=dict(id=r['author_id'], name=r['author_name'], kind=r['author_kind'])) for r in rows]
            coverage, notice = source.coverage_for(product, topic) if source.kind == 'incidents' else (source.coverage, source.notice)
            return dict(mode=source.kind, source_mode=source.mode, product=product, topic=topic or '', items=items,
                        **summary, topics=topics, topics_truncated=len(groups)>200, channels=channels,
                        coverage=coverage, notice=notice, snapshot_at=snapshot['shown_at'],snapshot_clock=snapshot['clock'],
                        current_cursor=self._encode(token), next_cursor=next_cursor, has_next=has_next, limit=limit)


class HistoryService:
    def __init__(self):
        self.reader = HistoryReader()
        self._lock, self._cache = threading.RLock(), {}

    def source(self, mode, supplied):
        with self._lock:
            if mode == 'incidents':
                if supplied is None:
                    return None
                key = (id(supplied),)
                create = lambda: LegacyHistory(supplied)
            elif isinstance(supplied, tuple):
                path, configured_mode = supplied; path = Path(path)
                key = (str(path), configured_mode, path.stat().st_ino if path.is_file() else None)
                create = lambda: NeedsHistory.from_path(path, configured_mode)
            else:
                key = (id(supplied),)
                create = lambda: NeedsHistory(supplied)
            previous = self._cache.get(mode)
            if previous is None or previous[0] != key:
                if previous:
                    previous[1].close()
                self._cache[mode] = (key, create())
            return self._cache[mode][1]


def install_history(app, legacy_source, needs_source, needs_principal):
    service = HistoryService()
    app.state.history_service = service

    def access(request, mode, product=None):
        if mode == 'needs':
            products = needs_principal(request, product)
            supplied = needs_source()
        else:
            # Existing incident middleware authenticates read access. Only the
            # products actually present in this source are readable here.
            supplied = legacy_source()
            products = sorted(supplied.rep.bins_total) if supplied is not None else []
            if product is not None and product not in products:
                raise HTTPException(403, 'Продукт недоступен в этом режиме')
        return products, service.source(mode, supplied)

    @app.get('/api/history/context')
    def context(request: Request, mode: Literal['incidents', 'needs']):
        products, source = access(request, mode)
        notice = (source.coverage_for(None, '')[1] if source and mode == 'incidents'
                  else source.notice if source else 'Инцидентный поток ещё не подготовлен. Откройте вкладку «Инциденты».')
        return dict(mode=mode, products=products, source_mode=source.mode if source else None,
                    notice=notice)

    @app.get('/api/history')
    def page(request: Request, mode: Literal['incidents', 'needs'], product: str = Query(min_length=1, max_length=96),
             topic: str = Query('', max_length=240), channel: str = Query('', max_length=64),
             start: str = Query('', max_length=40), end: str = Query('', max_length=40),
             limit: int = Query(50, ge=1, le=100), cursor: str | None = Query(None, max_length=8192)):
        _, source = access(request, mode, product)
        try:
            return service.reader.page(source, product=product, topic=topic, channel=channel, start=start, end=end, limit=limit, cursor=cursor)
        except SnapshotExpired as error:
            raise HTTPException(409, str(error)) from None
        except UnknownTopic as error:
            raise HTTPException(404, str(error)) from None
        except ValueError as error:
            raise HTTPException(400, str(error)) from None

    return service
