"""Persistent product-demand pipeline. Deterministic demo is not Qwen inference."""
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import math
import re
import sqlite3
import threading

from .privacy.mask import mask_text
from .privacy.mask_ner import available as ner_available

KINDS={"wish","bug","usability","service","unknown"}
DECISIONS={"accepted","known","rejected","needs_review","implemented"}
ID=re.compile(r"^[A-Za-z0-9_-]{1,96}$")
INJECTION=re.compile(r"игнорируй.{0,50}инструк|ignore.{0,50}instructions|покажи.{0,20}(секрет|промпт)|system\s*prompt",re.I|re.S)


def parse_time(value):
    t=datetime.fromisoformat(value.replace("Z","+00:00"))
    if t.tzinfo is None:
        raise ValueError("Timezone required")
    return t.astimezone(timezone.utc)


def demo_analyze(text, catalog):
    """Small, disclosed rule baseline; unrecognized texts stay unknown."""
    s=text.lower().replace("ё","е")
    rules=[
        (r"отключ.*уведом|не (?:нужны|присылайте).*уведом", "disable-notifications","Отключение уведомлений","wish"),
        (r"уведом|оповещ", "order-notifications","Уведомления о статусе заказа","wish"),
        (r"(?:экспорт|выгруз|excel).*(?:не работа|ошиб)|не (?:работает|получается).*(?:экспорт|выгруз)","export-failure","Ошибка экспорта заказов","bug"),
        (r"excel|выгруз|экспорт|таблиц.*заказ", "export-orders","Выгрузка истории заказов","wish"),
        (r"(?:телефон|номер).*(?:заявк|недоступ)|(?:измен|добав|замен|смен).*(?:телефон|контакт)","change-contact","Изменение контакта в заявке","service"),
        (r"смс|sms|код.*(?:вход|парол)","sms-login","Не приходит код входа","bug"),
        (r"очеред|404|время не существует","booking-queue","Бронирование электронной очереди","bug"),
        (r"машин|автомоб|справочник","vehicle-directory","Машина исчезла из справочника","bug"),
        (r"опф|ип вместо ооо|оферт.*(?:ип|предприниматель)","contract-legal-form","Неверная организационная форма в договоре","bug"),
        (r"кнопк|далее","contract-next-button","Кнопка продолжения договора не отвечает","bug"),
        (r"логин|договор уже заключен","restore-access","Восстановление доступа организации","service"),
        (r"непонят|не могу найти|где найти","navigation-help","Непонятная навигация","usability"),
    ]
    for pattern,key,label,kind in rules:
        if re.search(pattern,s):
            return dict(topic_key=key,label=label,kind=kind,quote=text[:300],confidence=.8)
    return dict(topic_key="unknown",label="Нужен разбор",kind="unknown",quote=text[:300],confidence=0)


def validate_analysis(a,text):
    if not isinstance(a,dict) or set(a)!={"topic_key","label","kind","quote","confidence"}:
        raise ValueError("Analysis schema")
    if a["kind"] not in KINDS or not isinstance(a["topic_key"],str) or not ID.fullmatch(a["topic_key"]):
        raise ValueError("Analysis taxonomy")
    if not isinstance(a["label"],str) or not 1<=len(a["label"])<=120:
        raise ValueError("Label length")
    if not isinstance(a["quote"],str) or not 1<=len(a["quote"])<=500 or a["quote"] not in text:
        raise ValueError("Evidence must be verbatim")
    c=a["confidence"]
    if isinstance(c,bool) or not isinstance(c,(float,int)) or not math.isfinite(c) or not 0<=c<=1:
        raise ValueError("Confidence range")
    return a


class Radar:
    def __init__(self,path,*,secret,now=None,analyzer=None,extractor=None,tracker=(),ner="auto",min_users=3,window_days=28):
        if not secret or len(secret)<16:
            raise ValueError("Use a persistent private HMAC key of at least 16 characters")
        self.secret=secret.encode();self.fixed_now=parse_time(now) if now else None
        self.analyzer=analyzer or demo_analyze; self.extractor=extractor
        self.mode="demo-rules" if analyzer is None else getattr(analyzer,"mode","internal-model")
        self.minimum_confidence=getattr(analyzer,"minimum_confidence",.65)
        self.ner=ner;self.min_users=min_users;self.window_days=window_days;self.tracker=tuple(tracker)
        self.lock=threading.RLock()
        self.db=sqlite3.connect(str(path),check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        PRAGMA secure_delete=ON;
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tickets(product TEXT,id TEXT,author TEXT,created TEXT,
            fingerprint TEXT,status TEXT,text TEXT,analysis TEXT,channel TEXT,PRIMARY KEY(product,id));
        CREATE INDEX IF NOT EXISTS tickets_window ON tickets(product,created,status);
        CREATE TABLE IF NOT EXISTS decisions(product TEXT,topic TEXT,status TEXT,updated TEXT,PRIMARY KEY(product,topic));
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY,at TEXT,product TEXT,kind TEXT,ref TEXT);
        ''')
        keytag=hmac.new(self.secret,b"radar-key-consistency",hashlib.sha256).hexdigest()
        existing=self.db.execute("SELECT value FROM settings WHERE key='keytag'").fetchone()
        if existing and not hmac.compare_digest(existing[0],keytag):
            self.db.close();raise ValueError("HMAC key differs from database key")
        self.db.execute("INSERT OR IGNORE INTO settings VALUES('keytag',?)",(keytag,))
        cutoff=(self.now()-timedelta(days=365)).isoformat()
        self.db.execute("DELETE FROM tickets WHERE created < ?",(cutoff,))
        self.db.commit()

    def now(self):
        return self.fixed_now or datetime.now(timezone.utc)

    def close(self):
        self.db.close()

    def _hash(self,value):
        return hmac.new(self.secret,value.encode(),hashlib.sha256).hexdigest()

    def ingest(self,t):
        with self.lock:
            return self._ingest(t)

    def _ingest(self,t):
        product,ident=t.get("product",""),t.get("id","")
        if not isinstance(product,str) or not isinstance(ident,str) or not ID.fullmatch(product) or not ID.fullmatch(ident):
            raise ValueError("Invalid product or id")
        author=t.get("author")
        if not isinstance(author,str) or not 1<=len(author)<=256:
            raise ValueError("Stable source author id required")
        created=parse_time(t["created_at"]); now=self.now()
        if not now-timedelta(days=365)<=created<=now:
            raise ValueError("Timestamp outside retained history")
        raw=t.get("text","")
        if not isinstance(raw,str) or len(raw)>4000:
            raise ValueError("Text exceeds 4000 chars; split upstream with shared parent id")
        channel=t.get("channel","text")
        if channel not in {"text","audio","image","document"}:
            raise ValueError("Unsupported channel")
        attachments=t.get('attachments',[])
        if not isinstance(attachments,list) or len(attachments)>5 or any(not isinstance(a,dict) for a in attachments):
            raise ValueError('At most five attachments')
        if channel=="text" and not raw.strip() and not attachments:
            raise ValueError("Empty ticket")
        # HMAC prevents easy dictionary lookup of personal identifiers and raw text.
        ah=self._hash(product+":"+author)
        fingerprint=self._hash(json.dumps(t,sort_keys=True,ensure_ascii=False))
        old=self.db.execute("SELECT fingerprint,status FROM tickets WHERE product=? AND id=?",(product,ident)).fetchone()
        if old and old[0]==fingerprint and old[1] not in {"pending_media","pending_analysis"}:
            return {"status":"duplicate","id":ident}
        if not old:
            count=self.db.execute("SELECT count(*) FROM tickets WHERE product=? AND author=? AND substr(created,1,10)=?",(product,ah,created.date().isoformat())).fetchone()[0]
            if count>=20:
                return {"status":"rate_limited","id":ident}
            if self.db.execute("SELECT count(*) FROM tickets").fetchone()[0]>=10000:
                raise ValueError("Demo capacity reached; use production storage")
        status="accepted"; clean=""; analysis={}
        media=([dict(t,channel=channel)] if channel!='text' else [])+attachments
        if media:
            if self.extractor is None:
                status="pending_media"
            else:
                try:
                    for attachment in media:
                        extracted=self.extractor(attachment.get('channel','document'),attachment)
                        if not isinstance(extracted,str) or not extracted.strip() or len(extracted)>3500:
                            raise ValueError("Unusable media extraction")
                        raw=(raw+"\n"+extracted).strip()
                        if len(raw)>4000:
                            raise ValueError("Combined text too long")
                except Exception:
                    status="pending_media"
        if status=="accepted":
            try:
                if self.ner and not ner_available():
                    raise ValueError("Required NER unavailable")
                clean,_=mask_text(raw,ner=self.ner,ctx_names=True)
                if INJECTION.search(clean):
                    status="quarantined"
                else:
                    analysis=validate_analysis(self.analyzer(clean,self.catalog(product)),clean)
                    analysis["label"],_=mask_text(analysis["label"],ner=self.ner,ctx_names=True)
                    if analysis["kind"]=="unknown" or analysis["confidence"]<self.minimum_confidence:
                        status="unknown"
            except Exception:
                # No raw text, provider body, credentials or traceback in persistent output.
                status="pending_analysis";analysis={}
        # Unextracted media remains in source system, not in this DB.
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO tickets VALUES(?,?,?,?,?,?,?,?,?)",
                            (product,ident,ah,created.isoformat(),fingerprint,status,clean,json.dumps(analysis,ensure_ascii=False),channel))
            self.db.execute("INSERT INTO events(at,product,kind,ref) VALUES(?,?,?,?)",(now.isoformat(),product,status,ident))
        return {"status":status,"id":ident,"analysis":analysis,"mode":self.mode}

    def catalog(self,product):
        rows=self.db.execute("SELECT analysis FROM tickets WHERE product=? AND status='accepted' ORDER BY created DESC LIMIT 500",(product,)).fetchall()
        out={}
        for row in rows:
            a=json.loads(row[0]);key=(a["kind"],a["topic_key"])
            if key not in out:
                out[key]=dict(topic_key=a["topic_key"],label=a["label"],kind=a["kind"])
            if getattr(self.analyzer,"supports_examples",False):
                examples=out[key].setdefault("examples",[])
                if a["quote"] not in examples and len(examples)<3:examples.append(a["quote"])
        return list(out.values())[:100]

    def digest(self,product):
        with self.lock:
            cutoff=(self.now()-timedelta(days=self.window_days)).isoformat()
            rows=self.db.execute("SELECT * FROM tickets WHERE product=? AND status='accepted' AND created>=? ORDER BY created,id",(product,cutoff)).fetchall()
            groups=defaultdict(list)
            for row in rows:
                a=json.loads(row["analysis"]);groups[(a["kind"],a["topic_key"])].append((row,a))
            cards=[]
            for (kind,key),items in groups.items():
                users={row["author"] for row,a in items}
                if len(users)<self.min_users:
                    continue
                topic=kind+":"+key
                decision=self.db.execute("SELECT status FROM decisions WHERE product=? AND topic=?",(product,topic)).fetchone()
                seen=set();evidence=[]
                for row,a in reversed(items):
                    if row["author"] not in seen and len(evidence)<5:
                        evidence.append({"id":row["id"],"product":product,"quote":a["quote"],"created_at":row["created"],"channel":row["channel"]});seen.add(row["author"])
                known=next((x["key"] for x in self.tracker if x["product"]==product and x["topic_key"]==key and x["kind"]==kind and x["status"] in {"open","in_progress"}),None)
                cards.append({"id":topic,"product":product,"title":items[0][1]["label"],"kind":kind,
                              "unique_users":len(users),"tickets":len(items),
                              "active_weeks":len({parse_time(row["created"]).strftime("%G-%V") for row,a in items}),
                              "first_seen":items[0][0]["created"],"last_seen":items[-1][0]["created"],
                              "evidence":evidence,"existing_task":known,
                              "hypothesis":kind=="service","status":decision[0] if decision else ("known" if known else "needs_review"),
                              "mode":self.mode,"window_days":self.window_days})
            return sorted(cards,key=lambda c:(-c["unique_users"],c["id"]))

    def decide(self,product,topic,status):
        if status not in DECISIONS:
            raise ValueError("Invalid decision")
        with self.lock:
            if topic not in {c["id"] for c in self.digest(product)}:
                raise KeyError("Card not found in product")
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?,?)",(product,topic,status,self.now().isoformat()))
                self.db.execute("INSERT INTO events(at,product,kind,ref) VALUES(?,?,?,?)",(self.now().isoformat(),product,"decision:"+status,topic))

    def stats(self):
        with self.lock:
            counts={r[0]:r[1] for r in self.db.execute("SELECT status,count(*) FROM tickets GROUP BY status")}
            return {"total":sum(counts.values()),"mode":self.mode,**counts}
