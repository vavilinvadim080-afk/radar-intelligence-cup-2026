"""Run local-only demo; no outbound service or message delivery by default."""
import argparse
import json
import os
from pathlib import Path
import secrets
import webbrowser
import uvicorn
from .core import Radar
from .gateway import InternalGateway
from .server import create_app

def main():
    p=argparse.ArgumentParser(description='Радар — официальный кейс')
    p.add_argument('--port',type=int,default=8796)
    p.add_argument('--mode',choices=['demo','internal'],default='demo')
    p.add_argument('--no-open',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    secrets_file=root/'data'/'.local-secrets.json'
    if not secrets_file.exists():
        fd=os.open(secrets_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump({'hmac':secrets.token_hex(32),'token':secrets.token_urlsafe(32)},f)
    cfg=json.loads(secrets_file.read_text())
    kw={}
    if args.mode=='internal':
        gateway=InternalGateway(os.environ['RADAR_MODEL_URL'],allowed_hosts=os.environ['RADAR_MODEL_IPS'].split(','),
            model=os.environ.get('RADAR_MODEL_NAME','Qwen 3.8'),asr_model=os.environ.get('RADAR_ASR_NAME','Qwen-ASR'),
            vl_model=os.environ.get('RADAR_VL_NAME','Qwen-VL'),token=os.environ.get('RADAR_MODEL_TOKEN',''))
        kw={'analyzer':gateway.analyze,'extractor':gateway.extract}
    else:kw={'now':'2026-11-01T12:00:00Z'}
    radar=Radar(root/'data'/f'{args.mode}.db',secret=cfg['hmac'],**kw)
    app=create_app(radar,tokens={cfg['token']:{'role':'admin','products':['webneft','portal']}})
    url=f'http://127.0.0.1:{args.port}/'
    print('Радар:',url,flush=True)
    print('Режим:',args.mode,'· ключ хранится только в data/.local-secrets.json',flush=True)
    if not args.no_open:
        import threading
        threading.Timer(1.5,lambda:webbrowser.open(url+'#token='+cfg['token'])).start()
    try:uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
    finally:radar.close()

if __name__=='__main__':main()
