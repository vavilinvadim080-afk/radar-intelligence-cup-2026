"""Explicit download of pinned public artifacts. The demo never calls this script."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES={'e5':'e5-small-v5b','qwen':'qwen-embedding-v5b','nli':'nli-minilm-v5b'}


def verified(path,meta):
    if not path.is_file() or path.stat().st_size!=meta['bytes']:return False
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()==meta['sha256']


def main():
    parser=argparse.ArgumentParser(description='Скачать закреплённые локальные модели; нужны сеть и несколько ГБ диска')
    parser.add_argument('models',nargs='+',choices=sorted(DIRECTORIES))
    args=parser.parse_args()
    for name in dict.fromkeys(args.models):
        directory=ROOT/'runtime/models'/DIRECTORIES[name]
        manifest=json.loads((directory/'MANIFEST.json').read_text())
        for filename,meta in manifest['files'].items():
            target=directory/filename
            if not target.resolve().is_relative_to(directory.resolve()):raise ValueError('Invalid artifact path')
            if verified(target,meta):
                print(name,filename,'уже проверен',flush=True);continue
            quoted='/'.join(urllib.parse.quote(p,safe='') for p in filename.split('/'))
            url='https://huggingface.co/'+manifest['model']+'/resolve/'+manifest['revision']+'/'+quoted
            temporary=target.with_name(target.name+'.part');target.parent.mkdir(parents=True,exist_ok=True)
            print(name,filename,'скачивание',meta['bytes'],'байт',flush=True)
            try:
                size=0
                with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:
                    while chunk:=response.read(1024*1024):
                        size+=len(chunk)
                        if size>meta['bytes']:raise ValueError('Artifact exceeds pinned size')
                        stream.write(chunk)
                if not verified(temporary,meta):raise ValueError('Artifact size or SHA256 mismatch')
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
    print('Выбранные артефакты проверены по размерам и SHA256.')


if __name__=='__main__':main()
