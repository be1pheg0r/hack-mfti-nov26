(() => {
  'use strict';
  const API_URL = '/api/dataset';
  const GROUPS_FALLBACK = [
    { id: 'it', label: 'IT и разработка' }, { id: 'finance', label: 'Финансы' },
    { id: 'education', label: 'Образование' }, { id: 'engineering', label: 'Инженерия' },
    { id: 'office_service', label: 'Офис и сервис' }
  ];
  const GROUP_LABELS = Object.fromEntries(GROUPS_FALLBACK.map(x => [x.id, x.label]));
  const state = { records: [], manifest: {}, groups: GROUPS_FALLBACK, kind: 'all', group: '', query: '', sort: 'source-date', favoritesOnly: false, favorites: new Set(), selectedId: '', tab: 'overview', theme: 'light', dataset: 'mini', limit: 100 };
  let loadSequence = 0;
  const $ = (s, root = document) => root.querySelector(s);
  const $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const safeStorage = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v == null ? fallback : v; } catch (_) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (_) {} }
  };
  const el = (tag, cls, text) => { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined) node.textContent = text; return node; };
  const recordById = id => state.records.find(r => String(r.id) === String(id));
  const kindLabel = kind => kind === 'resume' ? 'Резюме' : kind === 'vacancy' ? 'Вакансия' : 'Запись';
  const groupLabel = id => state.groups.find(g => g.id === id)?.label || GROUP_LABELS[id] || id || 'Не указано';
  const escText = value => value == null || value === '' ? 'Не указано' : String(value);
  const showToast = message => { const toast = $('#toast'); toast.textContent = message; toast.classList.add('show'); clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.classList.remove('show'), 2200); };
  const dateValue = r => r?.source?.published_at || '';
  const dateLabel = value => { if (!value) return 'Дата не указана'; const d = new Date(value); return Number.isNaN(+d) ? String(value) : new Intl.DateTimeFormat('ru-RU', { year: 'numeric', month: 'short', day: 'numeric' }).format(d); };
  const isFavorite = r => state.favorites.has(String(r.id));

  function filteredRecords() {
    let list = state.records.filter(r => (state.kind === 'all' || r.kind === state.kind) && (!state.group || r.sampling_group === state.group) && (!state.favoritesOnly || isFavorite(r)));
    const normalizeSearch = value => String(value).toLocaleLowerCase('ru').replace(/[-\s]+/g, ' ').trim();
    const q = normalizeSearch(state.query);
    if (q) list = list.filter(r => normalizeSearch(`${r.id} ${r.title || ''} ${r.text || ''} ${JSON.stringify(r.attributes || {})} ${r.sampling_group || ''}`).includes(q));
    list.sort((a, b) => {
      if (state.sort === 'title') return String(a.title || '').localeCompare(String(b.title || ''), 'ru');
      if (state.sort === 'length') return String(b.text || '').length - String(a.text || '').length;
      return String(dateValue(b)).localeCompare(String(dateValue(a))) || String(a.id).localeCompare(String(b.id));
    });
    return list;
  }
  function recordIcon(kind) { return kind === 'resume' ? 'Р' : 'В'; }
  function renderGroups() {
    const host = $('#group-options'); host.replaceChildren();
    const all = el('button', `group-option${state.group === '' ? ' selected' : ''}`); all.type = 'button'; all.dataset.group = ''; all.setAttribute('aria-pressed', String(!state.group));
    all.append(el('span', 'group-dot'), el('span', '', 'Все области'));
    const allCount = state.records.length; all.append(el('span', 'group-count', String(allCount))); host.append(all);
    state.groups.forEach(g => { const count = state.records.filter(r => r.sampling_group === g.id).length; const b = el('button', `group-option${state.group === g.id ? ' selected' : ''}`); b.type = 'button'; b.dataset.group = g.id; b.setAttribute('aria-pressed', String(state.group === g.id)); b.append(el('span', 'group-dot'), el('span', '', g.label), el('span', 'group-count', String(count))); host.append(b); });
  }
  function renderCounts() {
    $('#total-count').textContent = String(state.records.length);
    $('#count-all').textContent = String(state.records.length);
    $('#count-resume').textContent = String(state.records.filter(r => r.kind === 'resume').length);
    $('#count-vacancy').textContent = String(state.records.filter(r => r.kind === 'vacancy').length);
    const count = filteredRecords().length; $('#filtered-count').textContent = String(count);
    $('#list-summary').textContent = `${count} из ${state.records.length} записей`;
  }
  function renderList() {
    const host = $('#record-list'); host.replaceChildren();
    const records = filteredRecords();
    if (!records.length) {
      const empty = el('div', 'empty-list'); empty.append(el('strong', '', state.records.length ? 'Ничего не найдено' : 'Данные не загружены'), el('span', '', state.records.length ? 'Измените запрос или сбросьте фильтры.' : 'Проверьте, что сервер доступен и API запущен.'));
      const clear = el('button', '', 'Сбросить фильтры'); clear.type = 'button'; clear.addEventListener('click', clearFilters); empty.append(clear); host.append(empty); renderCounts(); return;
    }
    records.slice(0, state.limit).forEach(r => {
      const button = el('button', `record-option${String(r.id) === state.selectedId ? ' selected' : ''}`); button.type = 'button'; button.role = 'option'; button.dataset.id = r.id; button.setAttribute('aria-selected', String(String(r.id) === state.selectedId));
      const icon = el('span', `kind-icon${r.kind === 'vacancy' ? ' vacancy' : ''}`, recordIcon(r.kind)); icon.setAttribute('aria-hidden', 'true');
      const copy = el('span', 'record-copy'); copy.append(el('span', 'record-title', displayTitle(r)), el('span', 'record-meta', `${kindLabel(r.kind)} · ${groupLabel(r.sampling_group)} · ${dateLabel(dateValue(r))}`));
      button.append(icon, copy); if (isFavorite(r)) { const star = el('span', 'record-star', '★'); star.setAttribute('aria-label', 'В избранном'); button.append(star); }
      button.addEventListener('click', () => selectRecord(r.id)); host.append(button);
    });
    if (records.length > state.limit) { const more=el('button','load-more',`Показать ещё (${records.length - state.limit})`);more.type='button';more.addEventListener('click',()=>{state.limit+=100;renderList();});host.append(more); }
    renderCounts();
  }
  function displayTitle(r) {
    let title = String(r.title || 'Без названия');
    // Some copied resume titles separate Cyrillic words with hyphens; leave source intact.
    if (r.kind === 'resume') title = title.replace(/([А-Яа-яЁё])\s*-\s*(?=[А-Яа-яЁё])/g, '$1 ');
    return title;
  }
  function selectRecord(id, updateHash = true) {
    const r = recordById(id); if (!r) return;
    state.selectedId = String(r.id); state.tab = 'overview';
    const index = filteredRecords().findIndex(item => item.id === r.id); if (index >= state.limit) state.limit = index + 1;
    if (updateHash) history.replaceState(null, '', `#${encodeURIComponent(r.id)}`);
    $('#reader-empty').hidden = true; $('#reader-content').hidden = false;
    $('.browser').classList.add('show-reader'); $('#mobile-back').hidden = false;
    renderList(); renderDetail(r);
  }
  function addField(parent, label, value) { const box = el('div', 'meta-item'); box.append(el('dt', '', label)); const dd = el('dd', value == null || value === '' ? 'missing' : '', escText(value)); box.append(dd); parent.append(box); }
  function attrsOf(r) { return r.attributes && typeof r.attributes === 'object' ? r.attributes : {}; }
  const FIELD_LABELS = { experience: 'Опыт работы', experienceRequirements: 'Требуемый опыт', education: 'Образование', educationRequirements: 'Требования к образованию', scheduleType: 'График работы', busyType: 'Занятость', salaryMin: 'Зарплата от', salaryMax: 'Зарплата до', responsibilities: 'Обязанности', positionRequirements: 'Требования к должности', professionalArea: 'Профессиональная область', employment: 'Занятость' };
  function displayField(key, value) {
    if (Array.isArray(value)) return value.length ? value.map(v => typeof v === 'object' ? JSON.stringify(v) : String(v)).join(' · ') : 'Не указано';
    if (typeof value === 'object' && value !== null) return JSON.stringify(value);
    if ((key === 'experience' || key === 'experienceRequirements') && value === 0) return '0 лет · значение источника';
    if ((key === 'salaryMin' || key === 'salaryMax') && value !== '' && value != null) return `${value} · валюта в источнике не указана`;
    return escText(value);
  }
  function renderOverview(r, panel) {
    const attrs = attrsOf(r), text = String(r.text || '');
    const summary = el('section', 'summary-block'); summary.append(el('h3', 'section-heading', 'Текст записи')); const excerpt = el('div', 'summary-text', text ? (text.length > 1250 ? `${text.slice(0, 1250)}…` : text) : 'Не указано'); summary.append(excerpt); panel.append(summary);
    const skills = ['skills', 'hardSkills', 'softSkills'].filter(k => Array.isArray(attrs[k]));
    if (skills.length) skills.forEach(key => { const vals = attrs[key].filter(x => x != null && String(x).trim()); const section = el('section', 'summary-block'); section.append(el('h3', 'section-heading', key === 'hardSkills' ? 'Профессиональные навыки' : key === 'softSkills' ? 'Личные качества' : 'Навыки')); const chips = el('div', 'chips'); if (vals.length) vals.forEach(v => chips.append(el('span', `skill-chip${key === 'softSkills' ? ' soft' : ''}`, String(v)))); else chips.append(el('span', 'meta-item missing', 'Не указано')); section.append(chips); panel.append(section); });
    const visibleFields = Object.entries(attrs).filter(([k]) => !['skills', 'hardSkills', 'softSkills'].includes(k));
    const grid = el('div', 'field-grid');
    const known = new Set(visibleFields.map(([k]) => k));
    ['experience','experienceRequirements','education','educationRequirements','scheduleType','busyType','salaryMin','salaryMax','responsibilities','positionRequirements'].filter(k => !known.has(k)).forEach(k => visibleFields.push([k, null]));
    visibleFields.forEach(([key, value]) => addField(grid, FIELD_LABELS[key] || key, displayField(key, value))); panel.append(grid);
  }
  function sourceGrid(source, r) {
    const grid = el('dl', 'source-grid'); const fields = [['ID записи', r.id], ['Область выборки', groupLabel(r.sampling_group)], ['Издатель', source.publisher], ['Дата публикации', source.published_at], ['Исходный URL', source.original_url], ['Репозиторий копии', source.copy_repository], ['Ревизия копии', source.copy_revision], ['Файл копии', source.copy_file], ['Номер строки в источнике', source.source_row_zero_based], ['Тип ID источника', source.source_id_kind], ['ID источника', source.source_id], ['SHA-256 текста', r.text_sha256]];
    fields.forEach(([label, val]) => { const item = el('div', 'source-item'); item.append(el('dt', '', label), el('dd', '', escText(val))); grid.append(item); }); return grid;
  }
  function renderDetail(r) {
    const host = $('#reader-content'); host.replaceChildren(); const detail = el('article', 'record-detail');
    const topline = el('div', 'detail-topline'); const kind = el('div', 'detail-kind'); const ico = el('span', `kind-icon${r.kind === 'vacancy' ? ' vacancy' : ''}`, recordIcon(r.kind)); ico.setAttribute('aria-hidden', 'true'); kind.append(ico, el('span', '', kindLabel(r.kind)));
    const actions = el('div', 'detail-actions'); const fav = el('button', `action-button${isFavorite(r) ? ' favorite-active' : ''}`, isFavorite(r) ? '★ В избранном' : '♡ В избранное'); fav.type = 'button'; fav.setAttribute('aria-pressed', String(isFavorite(r))); fav.addEventListener('click', () => toggleFavorite(r.id));
    const copy = el('button', 'action-button', 'Копировать текст'); copy.type = 'button'; copy.addEventListener('click', () => copyText(String(r.text || ''), 'Текст скопирован'));
    const download = el('button', 'action-button', 'Скачать JSON'); download.type = 'button'; download.addEventListener('click', () => downloadJSON(r)); actions.append(fav, copy, download); topline.append(kind, actions); detail.append(topline);
    detail.append(el('h2', 'detail-title', displayTitle(r)), el('div', 'detail-id', `${r.id} · ${groupLabel(r.sampling_group)} · ${dateLabel(dateValue(r))}`), el('hr', 'detail-rule'));
    const tabs = el('div', 'tabs'); tabs.setAttribute('role', 'tablist'); [['overview','Обзор'],['text','Текст'],['json','JSON']].forEach(([id, label]) => { const b = el('button', 'tab-button', label); b.type = 'button'; b.setAttribute('role','tab'); b.setAttribute('aria-selected', String(state.tab === id)); b.id = `tab-${id}`; b.addEventListener('click', () => { state.tab = id; renderDetail(r); }); tabs.append(b); }); detail.append(tabs);
    const panel = el('div', 'tab-panel'); panel.id = 'panel'; panel.setAttribute('role','tabpanel'); panel.setAttribute('aria-labelledby', `tab-${state.tab}`);
    if (state.tab === 'overview') renderOverview(r, panel);
    else if (state.tab === 'text') { const tools = el('div','text-tools'); const b=el('button','action-button','Копировать текст');b.type='button';b.addEventListener('click',()=>copyText(String(r.text||''),'Текст скопирован'));tools.append(b);panel.append(tools, el('div','text-panel',String(r.text||'Не указано'))); if (Array.isArray(r.sections) && r.sections.length) { const section=el('section','summary-block');section.append(el('h3','section-heading','Секции источника')); const grid=el('div','field-grid'); r.sections.forEach(s=>{const label=`${escText(s.label)} · символы ${escText(s.start)}–${escText(s.end)}`;addField(grid,label, typeof s.source_field==='string'?s.source_field:'Не указано');});section.append(grid);panel.append(section); } }
    else { const tools=el('div','text-tools');const b=el('button','action-button','Копировать JSON');b.type='button';b.addEventListener('click',()=>copyText(JSON.stringify(r,null,2),'JSON скопирован'));tools.append(b);panel.append(tools,el('pre','json-panel',JSON.stringify(r,null,2))); }
    detail.append(panel); const source=el('details','source-card');source.append(el('summary','','Сведения об источнике и происхождении'));source.append(sourceGrid(r.source||{},r));detail.append(source);host.append(detail);
  }
  function copyText(text, message) { if (!text) { showToast('Нет текста для копирования'); return; } if (navigator.clipboard?.writeText) navigator.clipboard.writeText(text).then(() => showToast(message)).catch(() => fallbackCopy(text, message)); else fallbackCopy(text, message); }
  function fallbackCopy(text, message) { const area=el('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.append(area);area.select();try{document.execCommand('copy');showToast(message);}catch(_){showToast('Не удалось скопировать');}area.remove(); }
  function downloadJSON(record) { const blob=new Blob([JSON.stringify(record,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=el('a');a.href=url;a.download=`${String(record.id).replace(/[^\w.-]/g,'_')}.json`;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),500);showToast('JSON сохранён'); }
  const favoritesKey = () => state.dataset === 'mini' ? 'atlas-favorites' : `atlas-favorites-${state.dataset}`;
  function readFavorites() { try { const saved=JSON.parse(safeStorage.get(favoritesKey(),'[]')); state.favorites=new Set(Array.isArray(saved)?saved.map(String):[]); } catch (_) { state.favorites=new Set(); } }
  function toggleFavorite(id) { const key=String(id); if(state.favorites.has(key))state.favorites.delete(key);else state.favorites.add(key);safeStorage.set(favoritesKey(),JSON.stringify([...state.favorites]));renderList();const r=recordById(id);if(r)renderDetail(r); }
  function clearFilters() { state.query='';state.kind='all';state.group='';state.favoritesOnly=false;$('#search-input').value='';$$('.type-filter').forEach(b=>{const yes=b.dataset.kind==='all';b.classList.toggle('selected',yes);b.setAttribute('aria-pressed',String(yes));});$('#favorite-filter').setAttribute('aria-pressed','false');$('#favorite-filter').textContent='♡';renderGroups();renderList(); }
  function renderSourceDetails() {
    const m = state.manifest || {}, ranges = m.source_published_date_ranges || {}, prov = m.provenance || {};
    const host = $('#source-details'); host.replaceChildren();
    const grid = el('dl', 'source-grid');
    const fields = [
      ['Опубликовано (резюме)', ranges.resume ? `${dateLabel(ranges.resume.min)} — ${dateLabel(ranges.resume.max)}` : 'Не указано'],
      ['Опубликовано (вакансии)', ranges.vacancy ? `${dateLabel(ranges.vacancy.min)} — ${dateLabel(ranges.vacancy.max)}` : 'Не указано'],
      ['Исходный издатель', prov.original_publisher],
      ['Репозиторий копии', prov.copy_repository],
      ['Дата доступа, указанная копией', prov.original_access_date_reported_by_copy],
      ['Разметка парной релевантности', m.labels?.pair_relevance === false ? 'Отсутствует' : escText(m.labels?.pair_relevance)],
      ['Статусы требований', m.labels?.requirement_statuses === false ? 'Отсутствуют' : escText(m.labels?.requirement_statuses)]
    ];
    fields.forEach(([label, value]) => {
      const item = el('div', 'source-item');
      item.append(el('dt', '', label), el('dd', '', escText(value)));
      grid.append(item);
    });
    host.append(grid);
  }
  async function loadDataset() {
    const sequence=++loadSequence, dataset=state.dataset;
    const host=$('#record-list');host.replaceChildren();const loading=el('div','loading-state');loading.append(el('span','spinner'),el('span','','Загрузка локального набора…'));host.append(loading);
    try { const response=await fetch(`${API_URL}?dataset=${encodeURIComponent(dataset)}`,{headers:{Accept:'application/json'}});if(!response.ok)throw new Error(`Сервер вернул статус ${response.status}`);const data=await response.json();if(sequence!==loadSequence)return;if(!data||!Array.isArray(data.records))throw new Error('Ответ API не содержит список records');state.records=data.records;state.manifest=data.manifest||{};if(Array.isArray(data.groups)&&data.groups.length)state.groups=data.groups;$('#dataset-title').textContent=state.manifest.name||dataset;document.title=`${state.manifest.name||dataset} — ATLAS`;$('#dataset-version').textContent=`v ${state.manifest.version||'—'}`;$('#dataset-counts').textContent=`${state.manifest.actual?.resumes ?? '—'} · ${state.manifest.actual?.vacancies ?? '—'}`;renderGroups();renderSourceDetails();renderList();
      const hash=decodeURIComponent(location.hash.slice(1));if(hash&&recordById(hash))selectRecord(hash,false);else if(state.records.length){const first=filteredRecords()[0];if(first)selectRecord(first.id,false);}
    } catch(err) { if(sequence!==loadSequence)return;host.replaceChildren();const box=el('div','empty-list error-state');box.append(el('strong','','Не удалось загрузить данные'),el('span','',`${err.message}. Убедитесь, что локальный сервер запущен.`));const retry=el('button','','Повторить загрузку');retry.type='button';retry.addEventListener('click',loadDataset);box.append(retry);host.append(box);$('#list-summary').textContent='Нет соединения с API';$('#filtered-count').textContent='0'; }
  }
  async function loadCatalog() {
    try { const response=await fetch('/api/datasets'); if(!response.ok)throw new Error('catalog'); const data=await response.json(); const select=$('#dataset-select'); select.replaceChildren(); data.datasets.forEach(item=>{const option=el('option','',item.name);option.value=item.id;select.append(option);});select.value=state.dataset; }
    catch (_) { showToast('Не удалось загрузить список наборов'); }
  }
  function init() {
    readFavorites();
    state.theme=safeStorage.get('atlas-theme','light'); if(state.theme==='dark')document.documentElement.dataset.theme='dark';
    $('#search-input').addEventListener('input',e=>{state.query=e.target.value;renderList();});
    $$('.type-filter').forEach(b=>b.addEventListener('click',()=>{state.kind=b.dataset.kind;$$('.type-filter').forEach(x=>{const yes=x===b;x.classList.toggle('selected',yes);x.setAttribute('aria-pressed',String(yes));});renderList();}));
    $('#group-options').addEventListener('click',e=>{const b=e.target.closest('[data-group]');if(!b)return;state.group=b.dataset.group;renderGroups();renderList();});
    $('#sort-select').addEventListener('change',e=>{state.sort=e.target.value;renderList();});
    $('#favorite-filter').addEventListener('click',e=>{state.favoritesOnly=!state.favoritesOnly;e.currentTarget.setAttribute('aria-pressed',String(state.favoritesOnly));e.currentTarget.textContent=state.favoritesOnly?'★':'♡';renderList();});
    $('#clear-filters').addEventListener('click',clearFilters);
    $('#source-toggle').addEventListener('click',e=>{const box=$('#source-details');box.hidden=!box.hidden;e.currentTarget.setAttribute('aria-expanded',String(!box.hidden));});
    $('#mobile-back').addEventListener('click',()=>{$('.browser').classList.remove('show-reader');$('#mobile-back').hidden=true;});
    $('#theme-toggle').addEventListener('click',()=>{state.theme=state.theme==='dark'?'light':'dark';if(state.theme==='dark')document.documentElement.dataset.theme='dark';else delete document.documentElement.dataset.theme;safeStorage.set('atlas-theme',state.theme);$('#theme-toggle').setAttribute('aria-label',state.theme==='dark'?'Включить светлую тему':'Включить тёмную тему');});
    document.addEventListener('keydown',e=>{const tag=document.activeElement?.tagName;const typing=['INPUT','TEXTAREA','SELECT'].includes(tag)||document.activeElement?.isContentEditable;if(typing)return;if(e.key==='/'){e.preventDefault();$('#search-input').focus();}else if(e.key.toLowerCase()==='t'&&!e.ctrlKey&&!e.metaKey&&!e.altKey){$('#theme-toggle').click();}else if(e.key.toLowerCase()==='j'||e.key.toLowerCase()==='k'){const list=filteredRecords();if(!list.length)return;let idx=list.findIndex(r=>String(r.id)===state.selectedId);idx=e.key.toLowerCase()==='j'?Math.min(list.length-1,idx+1):Math.max(0,idx<0?0:idx-1);selectRecord(list[idx].id);}else if(e.key==='Escape'&&$('.browser').classList.contains('show-reader'))$('#mobile-back').click();});
    window.addEventListener('hashchange',()=>{const id=decodeURIComponent(location.hash.slice(1));if(id&&recordById(id))selectRecord(id,false);});
    $('#dataset-select').addEventListener('change',e=>{state.dataset=e.target.value;state.records=[];state.manifest={};state.selectedId='';state.limit=100;readFavorites();const url=new URL(location.href);url.searchParams.set('dataset',state.dataset);url.hash='';history.replaceState(null,'',url);$('#reader-content').hidden=true;$('#reader-empty').hidden=false;$('.browser').classList.remove('show-reader');$('#mobile-back').hidden=true;clearFilters();loadDataset();});
    state.dataset=new URLSearchParams(location.search).get('dataset')||'mini';readFavorites();
    renderGroups();loadCatalog();loadDataset();
  }
  init();
})();
