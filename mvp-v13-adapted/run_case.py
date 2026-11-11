"""Start the original dashboard with its added needs mode, bound to loopback."""
from pathlib import Path
import argparse
import os
import sys
import threading
import webbrowser
from urllib.parse import urlencode
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
os.environ.setdefault('RADAR_LAZY_INCIDENTS','1')
os.environ.setdefault('RADAR_MODE','template')
os.environ.setdefault('RADAR_AUDIT_DB',str(ROOT/'runtime'/'legacy-audit.db'))
from app.needs import private_config
cfg=private_config()
os.environ.setdefault('RADAR_API_TOKEN',cfg['token'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8797);p.add_argument('--no-open',action='store_true');a=p.parse_args()
    import uvicorn
    from app.server import app
    url=f'http://127.0.0.1:{a.port}/'
    print('Радар — инциденты и потребности:',url,flush=True)
    if not a.no_open:threading.Timer(1.5,lambda:webbrowser.open(url+'#'+urlencode({'token':os.environ['RADAR_API_TOKEN']}))).start()
    try:uvicorn.run(app,host='127.0.0.1',port=a.port,access_log=False)
    finally:
        r=getattr(app.state,'needs_radar',None)
        if r:r.close()
