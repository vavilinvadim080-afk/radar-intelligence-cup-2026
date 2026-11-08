from pathlib import Path
from html import escape
import re
ROOT=Path(__file__).resolve().parents[1]
def render(text):
    out = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('|'):
            table = []
            while i < len(lines) and lines[i].startswith('|'):
                row = lines[i]
                i += 1
                if re.fullmatch('[| :\\-]+', row):
                    continue
                table.append([escape(c.strip()) for c in row.strip('|').split('|')])
            out.append('<table><thead><tr>' + ''.join(('<th>' + c + '</th>' for c in table[0])) + '</tr></thead><tbody>')
            for row in table[1:]:
                out.append('<tr>' + ''.join(('<td>' + c + '</td>' for c in row)) + '</tr>')
            out.append('</tbody></table>')
            continue
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            out.append(f'<h{level}>' + escape(line.lstrip('# ')) + f'</h{level}>')
            i += 1
            continue
        out.append('<p>' + escape(line) + '</p>')
        i += 1
    return ''.join(out)
STYLE='@page{size:A4;margin:17mm 16mm}*{box-sizing:border-box}body{font:12pt/1.4 Arial,sans-serif;color:#183247;margin:0}h1{font-size:23pt;line-height:1.15;color:#153b59;margin:0 0 18pt}h2{font-size:16pt;margin:22pt 0 10pt;border-bottom:1px solid #c7dce6;padding-bottom:6pt;break-after:avoid}h3{font-size:13pt;break-after:avoid}p{margin:0 0 8pt;orphans:3;widows:3}table{border-collapse:collapse;width:100%;font-size:10pt;margin:10pt 0 16pt}th{background:#e1edf4;text-align:left}th,td{border:1px solid #adc6d6;padding:6pt;vertical-align:top}tr{break-inside:avoid}thead{display:table-header-group}'

if __name__=='__main__':
    for name in ['solution','effect']:
        p=ROOT/'submission'/(name+'.md'); text=p.read_text()
        if name=='solution' and len(text)>15000:raise ValueError('15000 character limit')
        document='<!doctype html><html lang="ru"><meta charset="utf-8"><title>'+escape(p.stem)+'</title><style>'+STYLE+'</style><body>'+render(text)+'</body></html>'
        p.with_suffix('.html').write_text(document)
    print('solution.html and effect.html rebuilt')
