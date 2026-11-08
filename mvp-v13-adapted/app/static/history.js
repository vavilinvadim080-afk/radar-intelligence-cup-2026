/* One saved-history view for incident and needs cards. All requests are local GETs. */
(function(){
'use strict';
class HistoryPager{
  constructor(load){this.load=load;this.filters={};this.cursors=[];this.index=0;this.data=null;this.busy=false;this.serial=0;this.controller=null;}
  cancel(){this.serial++;this.controller?.abort();this.busy=false;}
  reset(filters){this.cancel();this.filters={mode:filters.mode,product:filters.product,topic:filters.topic||'',channel:filters.channel||'',start:filters.start||'',end:filters.end||'',limit:filters.limit||50};this.cursors=[];this.index=0;this.data=null;return this._load(null,0);}
  next(){return !this.busy&&this.data?.has_next?this._load(this.data.next_cursor,this.index+1):Promise.resolve(null);}
  back(){return !this.busy&&this.index>0?this._load(this.cursors[this.index-1],this.index-1):Promise.resolve(null);}
  async _load(cursor,index){
    this.controller?.abort();this.controller=new AbortController();const serial=++this.serial;
    const filters={...this.filters},query=new URLSearchParams(filters);if(cursor)query.set('cursor',cursor);this.busy=true;
    try{
      const data=await this.load('/api/history?'+query,{signal:this.controller.signal});
      if(serial!==this.serial)return null;
      if(data.mode!==filters.mode||data.product!==filters.product)throw Error('Ответ истории относится к другому контексту');
      this.data=data;this.index=index;this.cursors[index]=data.current_cursor;return data;
    }catch(error){if(serial!==this.serial)return null;throw error;}
    finally{if(serial===this.serial)this.busy=false;}
  }
}
if(typeof module==='object'&&module.exports)module.exports={HistoryPager};
if(typeof document==='undefined')return;

const root=document.createElement('section');root.id='history-view';root.className='panel history-view';root.hidden=true;root.setAttribute('aria-label','История обращений');
root.innerHTML=`<div class="history-heading"><div><h2>Обращения · Газпром нефть</h2><p class="small" id="history-origin"></p></div><button id="history-close">Закрыть историю</button></div>
<form id="history-filters" class="history-toolbar">
<label>Режим<select id="history-mode"><option value="incidents">Инциденты</option><option value="needs">Потребности</option></select></label>
<label>Продукт<select id="history-product"></select></label>
<label>Канал<select id="history-channel"><option value="">Все каналы</option></select></label>
<label>С даты<input id="history-start" type="date"></label><label>По дату<input id="history-end" type="date"></label>
<label>На странице<select id="history-limit"><option value="50">50</option><option value="100">100</option></select></label>
<button type="button" id="history-refresh">Обновить историю</button></form>
<p id="history-notice" class="history-notice small"></p><p id="history-error" role="alert" hidden></p>
<div class="history-columns"><aside class="history-topics"><label class="small" for="history-search">Найти среди доступных тем</label><input id="history-search" type="search" placeholder="Название или ID темы"><div id="history-topics"></div></aside>
<div class="history-results"><div class="history-summary"><b id="history-topic-title">Все обращения</b><span id="history-total" class="small"></span></div><div id="history-messages" class="scroll" aria-live="polite"></div>
<div class="history-pages"><button id="history-prev">Назад</button><span id="history-page" class="small">Страница 1</span><button id="history-next">Далее</button></div></div></div>
<p class="history-foot small">История показывает сохранённые данные и не запускает анализ. Локальная очередь черновиков не означает отправку.</p>`;
document.body.append(root);
const el=id=>root.querySelector('#history-'+id),pager=new HistoryPager((path,options)=>api(path,options));
const names={webneft:'ВебНефть',bonus:'Бонусы',azs:'Приложение АЗС',portal:'Портал',b2b:'B2B-портал',partners:'Партнёры'};
const channelNames={text:'Текст',chat:'Чат',email:'Почта',audio:'Аудио',image:'Изображение',document:'Документ',review:'Отзыв'};
const statuses={accepted:'Обработано',unknown:'Нужен разбор',quarantined:'Карантин',pending_media:'Ожидает извлечения',pending_analysis:'Ожидает анализа',saved:'Сохранённая запись'};
let origin=null,contextSerial=0,contextPending=false;const scrolls=new Map();
function text(tag,value,className){const node=document.createElement(tag);node.textContent=value;if(className)node.className=className;return node;}
function locate(){root.style.top=document.querySelector('body > nav').getBoundingClientRect().bottom+'px';}
function error(message){el('error').textContent=message||'';el('error').hidden=!message;}
function controls(){const locked=pager.busy||contextPending;el('prev').disabled=locked||pager.index===0;el('next').disabled=locked||!pager.data?.has_next;el('refresh').disabled=locked||!el('product').value;el('product').disabled=contextPending;for(const id of ['channel','start','end','limit'])el(id).disabled=contextPending||!el('product').value;el('page').textContent=locked?'Загрузка…':'Страница '+(pager.index+1);}
function selectedFilters(){return {mode:el('mode').value,product:el('product').value,topic:pager.filters.topic||'',channel:el('channel').value,start:el('start').value,end:el('end').value,limit:+el('limit').value};}
async function run(action){
  if(pager.data)scrolls.set(pager.data.current_cursor,el('messages').scrollTop);
  error('');const promise=action();controls();
  try{const data=await promise;if(data)render(data);}catch(e){if(e.name!=='AbortError')error(e.message||String(e));}finally{controls();}
}
function reset(changes={}){scrolls.clear();return run(()=>pager.reset({...selectedFilters(),...changes}));}
function topicButton(title,id,detail){
  const button=document.createElement('button');button.type='button';button.className='history-topic'+((pager.filters.topic||'')===id?' on':'');
  button.setAttribute('aria-pressed',String((pager.filters.topic||'')===id));button.append(text('b',title),text('span',detail,'small'));
  button.onclick=()=>reset({topic:id});return button;
}
function renderTopics(data){
  const host=el('topics');host.replaceChildren(topicButton('Все обращения','','Вся доступная история продукта'));
  const search=el('search').value.trim().toLowerCase();
  for(const topic of data.topics){
    if(search&&!`${topic.title} ${topic.id}`.toLowerCase().includes(search))continue;
    host.append(topicButton(topic.title,topic.id,`${topic.total} обращений · ${topic.unique_authors===null?'ID авторов недоступны':topic.unique_authors+' разных авторов'}`));
  }
  if(data.topics_truncated)host.append(text('p','Показаны первые 200 тем. Нужную тему можно открыть из карточки.','small'));
}
function render(data){
  el('notice').textContent=data.notice;
  const selected=data.topics.find(t=>t.id===data.topic);el('topic-title').textContent=selected?.title||data.topic||'Все обращения';
  el('total').textContent=`${data.total} обращений · ${data.unique_authors===null?'ID авторов сохранены не для всех сообщений':data.unique_authors+' разных авторов'} · снимок (${data.snapshot_clock}) ${data.snapshot_at.replace('T',' ').replace('+00:00',' UTC')}`;
  const selectedChannel=el('channel').value;el('channel').replaceChildren(new Option('Все каналы',''));
  for(const channel of data.channels)el('channel').append(new Option(channelNames[channel]||channel,channel));el('channel').value=selectedChannel;
  if(selectedChannel&&!data.channels.includes(selectedChannel)){el('channel').append(new Option(channelNames[selectedChannel]||selectedChannel,selectedChannel));el('channel').value=selectedChannel;}
  renderTopics(data);const messages=el('messages');messages.replaceChildren();
  for(const row of data.items){
    const item=document.createElement('article');item.className='card history-message';item.dataset.messageId=row.id;
    const meta=document.createElement('div');meta.className='history-message-meta';
    meta.append(text('code',row.id),text('span',row.created_at.replace('T',' ').replace('+00:00',' UTC')),text('span',channelNames[row.channel]||row.channel,'chip'),text('span',statuses[row.status]||row.status,'small'));
    const author=document.createElement('div');author.className='history-author small';
    if(row.author.name)author.append(text('span',row.author.name+' · '));
    if(row.author.id){
      const anonymized=row.author.kind==='hmac',ident=row.author.id;
      const code=text('code',anonymized?ident.slice(0,12)+'…'+ident.slice(-4):ident);
      if(anonymized)code.title='Полный обезличенный код: '+ident;
      author.append(text('span',anonymized?'Обезличенный автор: ':'ID автора: '),code);
    }
    else author.append(text('span','ID автора не сохранён в источнике'));
    const body=text('p',row.text||'Текст ещё не извлечён или недоступен в сохранённом источнике.');
    item.append(meta,author,body);
    if(row.original_text!==null){const detail=document.createElement('details');detail.append(text('summary','Исходный текст из записи'),text('p',row.original_text||'Пустой текст источника'));item.append(detail);}
    else item.append(text('p','Оригинал в этой базе не сохраняется.','small'));
    const source=text('div','Источник: '+row.id,'small');
    if(row.source_url&&/^https?:\/\//i.test(row.source_url)){const link=text('a','Открыть источник');link.href=row.source_url;link.target='_blank';link.rel='noopener noreferrer';source.append(' · ',link);}
    else source.append(' · внешняя ссылка не передана');
    item.append(source);messages.append(item);
  }
  if(!data.items.length)messages.append(text('p',data.coverage==='confirmed_ids_only'?'Нет обращений по выбранной теме и фильтрам. Сохранённая подборка инцидента может быть неполной.':'Нет обращений по выбранной теме и фильтрам. История появится после загрузки источника.','pending'));
  messages.scrollTop=scrolls.get(data.current_cursor)||0;
}
async function sourceContext(mode,product,topic=''){
  const serial=++contextSerial;contextPending=true;pager.cancel();pager.data=null;pager.cursors=[];pager.index=0;pager.filters={mode,product:product||'',topic:''};error('');el('messages').replaceChildren(text('p','Загрузка истории…','pending'));el('topics').replaceChildren();el('topic-title').textContent='Все обращения';el('total').textContent='';el('mode').value=mode;el('product').replaceChildren();controls();
  try{
    const context=await api('/api/history/context?mode='+encodeURIComponent(mode));if(serial!==contextSerial)return;
    for(const value of context.products)el('product').append(new Option(names[value]||value,value));
    el('notice').textContent=context.notice;
    if(!context.products.length){el('messages').replaceChildren(text('p',context.notice,'pending'));el('topics').replaceChildren();return;}
    if(product&&!context.products.includes(product)){el('product').value='';throw Error('Продукт этой карточки недоступен в выбранном режиме');}
    el('product').value=product||context.products[0];el('channel').replaceChildren(new Option('Все каналы',''));el('start').value='';el('end').value='';el('search').value='';scrolls.clear();
    await run(()=>pager.reset({...selectedFilters(),mode,product:el('product').value,topic}));
  }catch(e){if(serial===contextSerial)error(e.message||String(e));}
  finally{if(serial===contextSerial){contextPending=false;controls();}}
}
function open(options={}){
  origin=options.topic?{...options}:null;root.hidden=false;locate();
  const currentMode=new URLSearchParams(location.search).get('mode')==='needs'?'needs':'incidents';
  el('origin').textContent=origin?'Из карточки '+origin.topic+' · '+(names[origin.product]||origin.product):'Сохранённая история двух режимов';
  el('close').textContent=origin?'Вернуться к карточке':'Закрыть историю';
  document.querySelector('#history-tab').setAttribute('aria-selected','true');el('close').focus();sourceContext(options.mode||currentMode,options.product,options.topic||'');
}
function close(){
  contextSerial++;pager.cancel();root.hidden=true;document.querySelector('#history-tab').setAttribute('aria-selected','false');
  if(origin){
    if(origin.mode==='needs')window.RadarNeeds?.select(origin.topic);
    const selector=origin.mode==='needs'?'.card[data-topic]':'.card[data-id]';
    const card=[...document.querySelectorAll(selector)].find(e=>(e.dataset.topic||e.dataset.id)===origin.topic);
    if(card){card.tabIndex=-1;card.scrollIntoView({block:'center'});card.focus();}
    else toast('Карточка уже недоступна в текущем состоянии источника');
  }else document.querySelector('#history-tab').focus();
}
document.querySelector('#history-tab').onclick=()=>open();el('close').onclick=close;
el('filters').onsubmit=e=>{e.preventDefault();reset();};
el('mode').onchange=()=>sourceContext(el('mode').value);
el('product').onchange=()=>reset({product:el('product').value,topic:''});
for(const id of ['channel','start','end','limit'])el(id).onchange=()=>reset();
el('search').oninput=()=>{if(pager.data)renderTopics(pager.data);};
el('refresh').onclick=()=>reset();el('prev').onclick=()=>run(()=>pager.back());el('next').onclick=()=>run(()=>pager.next());
window.addEventListener('resize',()=>{if(!root.hidden)locate();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!root.hidden){e.preventDefault();close();}});
document.addEventListener('click',e=>{const button=e.target.closest('[data-history-mode]');if(!button)return;e.preventDefault();e.stopPropagation();open({mode:button.dataset.historyMode,product:button.dataset.historyProduct,topic:button.dataset.historyTopic});},true);
window.RadarHistory={open,close};
})();
