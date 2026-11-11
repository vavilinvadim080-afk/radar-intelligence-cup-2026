/* Additional mode for the original dashboard. Uses the same controls and card styles. */
(async function(){
if(new URLSearchParams(location.search).get('mode')!=='needs')return;
let prod='webneft',selected=null,cards=[],context={},busy=false;
const kind={wish:'пожелание',bug:'ошибка',usability:'неудобство интерфейса',service:'гипотеза сервиса'},labels={needs_review:'нужно решение',accepted:'подтверждено',known:'уже известно',rejected:'отклонено'};
const showError=e=>{toast(e.message||String(e));$('#needs-error').textContent=e.message||String(e);};
async function load(){
 if(busy)return;busy=true;
 try{
  const status=await api('/api/needs/status');
  const select=$('#needs-product');
  if(!select.options.length){status.products.forEach(p=>{const o=document.createElement('option');o.value=p;o.textContent={webneft:'ВебНефть',portal:'Внутренний портал'}[p]||p;select.append(o)});prod=select.value;}
  const data=await api('/api/needs/topics?product='+encodeURIComponent(prod));cards=data.cards;
  context=await api('/api/needs/context?product='+encodeURIComponent(prod));
  $('#needs-mode').textContent=({'demo-rules':'Синтетика · правила, без модели','local-semantic-prototype':'Синтетика · локальная семантика','local-calibrated-v5b-experiment':'Синтетика · смысловой отбор, эксперимент','local-verifier-v5c-experiment':'Qwen · проверка похожих обращений, эксперимент','internal':'Внутренняя модель','internal-model':'Внутренняя модель'})[status.mode]||'Режим анализа не распознан';
  $('#needs-window').textContent=status.window_days+' дней · от '+status.min_users+' авторов';
  $('#needs-error').textContent='';
  $('#needs-count').textContent=cards.length+' тем';
  if(!cards.some(c=>c.id===selected))selected=cards[0]?.id||null;
  drawCards();drawDetail();drawSource();drawGraph();
  $('#needs-audit').innerHTML=(context.events||[]).slice(0,12).map(e=>`<div class="r"><span class="t">${esc((e.at||'').slice(11,16))}</span><span class="m">${esc(({accepted:'Обращение обработано',duplicate:'Повтор загрузки',quarantined:'Карантин',unknown:'Нужен разбор',pending_media:'Ожидает извлечения',pending_analysis:'Ожидает анализа','decision:accepted':'Потребность подтверждена','decision:known':'Уже известно','decision:rejected':'Отклонено'})[e.kind]||e.kind)}<br><small>${esc(e.ref)}</small></span></div>`).join('')||'<p class="small">Решения появятся после обработки.</p>';
  $('#needs-outbox').innerHTML=(context.outbox||[]).map(o=>`<div class="card"><b>${o.kind==='draft'?'Черновик задачи':'Дайджест'}</b><p class="small">${esc(o.id)} · ожидает доставки</p></div>`).join('')||'<p class="small">Черновиков пока нет. Реальная доставка не подключена.</p>';
 }catch(e){showError(e);}finally{busy=false;}
}
function drawCards(){
 const host=$('#needs-cards');host.innerHTML='';
 if(!cards.length){host.innerHTML='<div class="card"><div class="title">Пока нет повторных тем</div><p>Загрузите пример. Потребность появится после обращений трёх разных авторов.</p></div>';return;}
 cards.forEach(c=>{
  const el=document.createElement('article');el.className='card '+(c.id===selected?'alert':'watch');el.tabIndex=0;el.dataset.topic=c.id;
  el.style.cursor='pointer';
  el.innerHTML=`<div class="top"><span class="chip ${c.hypothesis?'warn':'blue'}">${esc(kind[c.kind]||c.kind)}</span><span class="chip ${c.status==='accepted'?'ok':''}">${esc(labels[c.status]||c.status)}</span></div><div class="title">${esc(c.title)}</div><div class="facts"><div class="fact"><b>${c.unique_users}</b><span>разных авторов</span></div><div class="fact"><b>${c.tickets}</b><span>обращений</span></div><div class="fact"><b>${c.active_weeks}</b><span>активных недель</span></div></div><p class="small">${esc(c.evidence[0]?.quote||'')} <span class="ref">${esc(c.evidence[0]?.id||'')}</span></p>`;
  const check=document.createElement('button');check.textContent='Перейти к проверке';check.dataset.historyMode='needs';check.dataset.historyProduct=prod;check.dataset.historyTopic=c.id;el.append(check);
  const choose=()=>{selected=c.id;drawCards();drawDetail();};el.onclick=choose;el.onkeydown=e=>{if(e.key==='Enter'&&e.target===el){e.preventDefault();choose();}};host.append(el);
 });
}
function drawDetail(){
 const c=cards.find(c=>c.id===selected),host=$('#needs-detail');host.innerHTML='';if(!c){host.innerHTML='<p class="small">Выберите тему в центре.</p>';return;}
 const box=document.createElement('div');
 box.innerHTML=`<div class="title" style="font-size:20px;font-weight:700">${esc(c.title)}</div><p class="small">${esc(kind[c.kind])} · ${esc(labels[c.status]||c.status)}</p><ul class="claims">${c.evidence.map(q=>`<li>«${esc(q.quote)}»<br><span class="ref">${esc(q.id)}</span> <span class="small">${esc(q.created_at.slice(0,10))}</span></li>`).join('')}</ul><p class="small">${c.hypothesis?'Гипотеза о новом сервисе требует исследования спроса.':'Цитаты относятся к разным авторам. Счётчики рассчитаны кодом.'}</p>`;host.append(box);
 const buttons=document.createElement('div');buttons.className='btns';host.append(buttons);
 [['accepted','Подтвердить','green'],['known','Уже известно',''],['rejected','Отклонить','']].forEach(([status,title,cl])=>{
  const b=document.createElement('button');b.textContent=title;b.className=cl;b.onclick=async()=>{b.disabled=true;try{await post('/api/needs/decision',{product:prod,topic:c.id,status});toast('Решение сохранено');await load();}catch(e){showError(e);b.disabled=false;}};buttons.append(b);
 });
 const draft=document.createElement('button');draft.textContent='Подготовить задачу';draft.className='primary';draft.disabled=c.status!=='accepted';draft.title=c.status==='accepted'?'Черновик останется в локальной очереди':'Сначала подтвердите потребность';draft.onclick=()=>post('/api/needs/tool',{product:prod,name:'queue_draft',args:{topic:c.id}}).then(()=>{toast('Черновик сохранён, отправки нет');load();}).catch(showError);host.append(draft);draft.style.marginTop='12px';
}
function drawSource(){
 $('#needs-source').innerHTML=(context.tickets||[]).slice(0,18).map(t=>`<div class="mbox" style="margin-bottom:8px"><h3>${esc(t.id)} · ${esc(({accepted:'обработано',unknown:'нужен разбор',quarantined:'карантин',pending_media:'ожидает извлечения',pending_analysis:'ожидает анализа'})[t.status]||t.status)}</h3><p>${esc(t.text||'Содержимое не извлечено — ожидает обработки')}</p></div>`).join('')||'<p class="small">Источники появятся после загрузки.</p>';
}
function drawGraph(){
 const svg=$('#needs-chart'),groups={};for(const t of context.tickets||[]){if(t.status==='accepted'){const day=t.created.slice(0,10);groups[day]=(groups[day]||0)+1;}}
 const dates=Object.keys(groups).sort(),values=dates.map(d=>groups[d]),w=580,h=200,m=36,max=Math.max(1,...values),xp=i=>m+i*(w-2*m)/Math.max(1,dates.length-1),yp=v=>h-m-v*(h-2*m)/max;
 let body='';for(let i=0;i<=4;i++){const v=Math.round(max*i/4),y=yp(v);body+=`<line x1="${m}" x2="${w-m}" y1="${y}" y2="${y}" stroke="var(--grid)"/><text x="8" y="${y+4}" fill="var(--muted)" font-size="12">${v}</text>`;}
 const pts=values.map((v,i)=>`${xp(i)},${yp(v)}`);if(pts.length){body+=`<polyline points="${pts.join(' ')}" fill="none" stroke="var(--accent)" stroke-width="3"/>`;dates.forEach((d,i)=>{body+=`<circle cx="${xp(i)}" cy="${yp(values[i])}" r="4" fill="var(--accent)"/><text x="${xp(i)}" y="${h-8}" text-anchor="middle" fill="var(--muted)" font-size="11">${esc(d.slice(5))}</text>`;});}
 svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.innerHTML=body;
}
$('#needs-product').onchange=e=>{prod=e.target.value;selected=null;load();};
window.RadarNeeds={select(topic){if(!cards.some(c=>c.id===topic))return false;selected=topic;drawCards();drawDetail();return true;}};
$('#needs-load').onclick=async()=>{const b=$('#needs-load');b.disabled=true;try{await post('/api/needs/demo',{});await load();toast('Пример загружен. Повторная загрузка не удваивает данные.');}catch(e){showError(e);}finally{b.disabled=false;}};
$('#needs-refresh').onclick=load;$('#needs-theme').onclick=()=>$('#theme').click();
$('#needs-help').onclick=()=>{$('#needs-explain').hidden=!$('#needs-explain').hidden;};
await load();
})();
