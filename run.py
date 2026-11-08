"""Portable launcher for the submission; binds only to the local machine."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description='Радар: инциденты и потребности')
    parser.add_argument('--port',type=int,default=8799,help='Локальный порт, по умолчанию 8799')
    parser.add_argument('--no-open',action='store_true',help='Показать адрес для входа вместо открытия браузера')
    parser.add_argument('--check',action='store_true',help='Проверить наличие зависимостей без запуска сервера')
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('Порт должен быть от 1024 до 65535')
    required=['fastapi','uvicorn','pydantic','numpy','scipy','sklearn','jsonschema','natasha','httpx']
    missing=[name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(1,'Не найдены пакеты: '+', '.join(missing)+'\nУстановите: python -m pip install -r requirements.txt\n')
    if args.check:
        print('Все зависимости локального демо найдены. Модели и сервер не запускались.')
        return
    os.environ.setdefault('RADAR_SEMANTIC_PYTHON',sys.executable)
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
    sys.path[:0]=[str(ROOT/'mvp-v13-adapted'),str(ROOT)]
    if args.no_open:
        from app.needs import private_config
        config=private_config()
        print(f'Локальный адрес для входа: http://127.0.0.1:{args.port}/#token={config["token"]}',flush=True)
        print('Адрес содержит локальный токен доступа. Не публикуйте его.',flush=True)
    sys.argv=[str(ROOT/'mvp-v13-adapted/run_case.py'),'--port',str(args.port)]
    if args.no_open:sys.argv.append('--no-open')
    runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
