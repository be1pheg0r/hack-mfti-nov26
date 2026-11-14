"use strict";
(function (root) {
  // A stage shows only values produced by that stage; an absent score is not zero.
  function stageRows(data, stage) {
    const gate = new Set([...data.gate.passed, ...(data.gate.unknown || [])]);
    const bm25 = new Map(data.bm25.map(item => [item.vacancy_id, item.score]));
    if (stage === 3) return data.ranked.map(item => ({id: item.vacancy_id, score: item.score, removed: false}));
    return data.initial.map(id => ({id, ...(stage >= 2 && bm25.has(id) ? {score: bm25.get(id)} : {}), removed: stage === 1 ? !gate.has(id) : stage === 2 ? !gate.has(id) || !bm25.has(id) : false}));
  }
  if (typeof module !== "undefined") module.exports = {stageRows};
  if (!root.document) return;
  const $ = selector => document.querySelector(selector);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);
  const state = {token: sessionStorage.getItem("svodka-token") || "", catalog: null, selected: null, result: null, stage: -1, busy: false, request: 0};
  const details = [
    {tag:"ТОЧКА ОТСЧЁТА", title:"Начнём с резюме", description:"Выберите кандидата — и посмотрите, как список вакансий превращается в рекомендацию.", lead:"Хороший поиск начинается с контекста.", body:"Резюме и вакансии представлены как документы с извлечёнными признаками. Система сопоставляет их поэтапно; отсутствие сведений не означает отсутствие навыка.", source:'<a href="https://doi.org/10.1561/1500000019" target="_blank" rel="noopener noreferrer"><b>NOW</b><span>Исследование информационного поиска ↗<small>Robertson & Zaragoza, 2009</small></span></a>'},
    {tag:"ИСХОДНЫЙ СПИСОК", title:"До начала отбора", description:"Это случайный порядок, а не оценка соответствия.", lead:"Сначала все вакансии на равных.", body:"Начальный порядок перемешан для демонстрации. Место в этом списке ничего не говорит о пригодности кандидата и не является прогнозом.", source:'<a href="https://doi.org/10.1561/1500000019" target="_blank" rel="noopener noreferrer"><b>NOW</b><span>Основы ранжирования ↗<small>Robertson & Zaragoza, 2009</small></span></a>'},
    {tag:"GATE 1", title:"Проверяем условия", description:"Отсеиваем явные несовместимости. Неизвестное — не отрицание.", lead:"Ограничения до ранжирования.", body:"Первый фильтр сопоставляет структурированные условия: опыт, образование, формат и график. Явные противоречия исключают вакансию; нехватка сведений не превращается автоматически в отказ.", source:'<a href="https://www.nist.gov/itl/ai-risk-management-framework" target="_blank" rel="noopener noreferrer"><b>NIST</b><span>AI Risk Management Framework ↗<small>Управление рисками ИИ</small></span></a>'},
    {tag:"GATE 2 / BM25", title:"Ищем по словам", description:"BM25 оставляет ограниченное число самых релевантных вакансий.", lead:"Быстрый поиск по важным словам.", body:"BM25 сопоставляет термины и учитывает их частоту и распространённость. На следующий этап проходит ограниченное конфигурацией число вакансий; оценка BM25 — не вероятность трудоустройства.", source:'<a href="https://doi.org/10.1561/1500000019" target="_blank" rel="noopener noreferrer"><b>NOW</b><span>BM25 and Beyond ↗<small>Robertson & Zaragoza, 2009</small></span></a>'},
    {tag:"EMBEDDINGGEMMA 2", title:"Сравниваем смысл", description:"Модель переставляет прошедшие вакансии по близости представлений.", lead:"Не только точные совпадения слов.", body:"EmbeddingGemma 2 преобразует тексты в векторы. Близость представлений помогает упорядочить короткий список; это мера похожести, а не гарантия соответствия. Итог требует проверки человеком.", source:'<a href="https://huggingface.co/google/embeddinggemma-2" target="_blank" rel="noopener noreferrer"><b><svg aria-label="Google" role="img" width="24" height="24" viewBox="0 0 24 24"><path fill="currentColor" d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z"/></svg></b><span>EmbeddingGemma 2 ↗<small>Google · карточка модели</small></span></a>'},
  ];
  async function api(path, options = {}) {
    const bytes = new TextEncoder().encode(state.token);
    const token = btoa(Array.from(bytes, byte => String.fromCharCode(byte)).join("")).replace(/\+/g,"-").replace(/\//g,"_").replace(/=+$/,"");
    const response = await fetch(path, {...options, headers: {Authorization:`Bearer utf8.${token}`, ...(options.body ? {"Content-Type":"application/json"} : {})}});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Ошибка соединения");
    return payload;
  }
  function explain(index) {
    const item = details[index];
    $("#explain-index").textContent = `ПРИНЦИП / ${String(index + 1).padStart(2,"0")}`;
    $("#explain-title").textContent = item.title + ".";
    $("#explain-lead").textContent = item.lead;
    $("#explain-body").textContent = item.body;
    $("#sources").innerHTML = item.source;
  }
  function choice() {
    const resumes = state.catalog.resumes;
    $("#resume-count").textContent = String(resumes.length).padStart(2,"0");
    $("#resume-list").innerHTML = resumes.map((r,i) => `<button class="resume-choice" data-id="${esc(r.id)}" aria-pressed="${r.id === state.selected}"><span>${String(i+1).padStart(2,"0")}</span><span><strong>${esc(r.name)}</strong><small>${esc(r.title)}${r.synthetic ? " · синтетическое" : ""}</small></span><b>↗</b></button>`).join("");
    const index = resumes.findIndex(r => r.id === state.selected), person = resumes[index];
    $("#preview-number").textContent = `РЕЗЮМЕ / ${String(index+1).padStart(2,"0")}`;
    $("#preview-name").textContent = person.name; $("#preview-title").textContent = person.title; $("#preview-text").textContent = person.text;
  }
  function setStage(stage) {
    state.stage = stage;
    const detail = details[stage + 1];
    $("#section-label").textContent = `0${stage+2} / ${detail.tag}`;
    $("#heading").textContent = detail.title + ".";
    $("#subtitle").textContent = state.catalog.resumes.find(r => r.id === state.selected)?.name || "";
    $("#stage-description").textContent = detail.description;
    $("#stages").innerHTML = details.slice(1).map((d,i) => `<li class="${i === stage ? "active" : i < stage ? "done" : ""}"><span>${String(i+1).padStart(2,"0")}</span><strong>${esc(d.tag)}</strong><b>${i < stage ? "✓" : i === stage ? "↗" : ""}</b></li>`).join("");
    const rows = stageRows(state.result,stage), lookup = new Map(state.catalog.vacancies.map(v => [v.id,v]));
    $("#table-title").textContent = stage === 3 ? "ИТОГОВЫЙ РЕЙТИНГ" : "ЛИДЕРБОРД ВАКАНСИЙ";
    $("#table-count").textContent = `${rows.filter(r => !r.removed).length} / ${rows.length}`;
    $("#list-hint").textContent = ["СЛУЧАЙНЫЙ ПОРЯДОК · НЕ ОЦЕНКА", "ЯВНЫЕ НЕСОВМЕСТИМОСТИ ВЫЧЁРКНУТЫ", "В ШОРТ-ЛИСТЕ — ЛУЧШИЕ ПО BM25", "СХОЖЕСТЬ ВЕКТОРОВ · НЕ ВЕРОЯТНОСТЬ"][stage];
    $("#jobs").innerHTML = rows.map((r,i) => `<div class="job-row ${r.removed ? "removed" : ""}"><span class="job-number">${String(i+1).padStart(2,"0")}</span><span class="job-name">${esc(lookup.get(r.id)?.title || r.id)}</span><span class="job-score">${r.removed ? "ОТСЕЯНА" : r.score == null ? "—" : Number(r.score).toFixed(3)}</span></div>`).join("") || '<p class="empty">На этом этапе не осталось вакансий.</p>';
    $("#next").hidden = stage === 3;
    $("#next").disabled = state.busy;
    $("#next").innerHTML = stage === 2 ? "К РЕЙТИНГУ <span>↗</span>" : "ДАЛЬШЕ <span>↗</span>";
    explain(stage + 1);
  }
  async function enter() {
    try { state.catalog = await api("/api/demo"); if (!state.catalog.resumes.length) throw new Error("Резюме не найдены."); state.selected = state.catalog.resumes[0].id; sessionStorage.setItem("svodka-token",state.token); $("#login").hidden = true; $("#app").hidden = false; $("#logout").hidden = false; choice(); explain(0); }
    catch (error) { $("#login-error").textContent = error.message; }
  }
  $("#login-form").addEventListener("submit", event => {event.preventDefault(); state.token = $("#token").value; enter();});
  $("#logout").addEventListener("click", () => {sessionStorage.removeItem("svodka-token"); location.reload();});
  $("#resume-list").addEventListener("click", event => {const button = event.target.closest("[data-id]"); if (button) {state.selected = button.dataset.id; choice();}});
  $("#analyze").addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true; const request = ++state.request;
    $("#choose").hidden = true; $("#journey").hidden = false; $("#error").textContent = "";
    const initial = state.catalog.vacancies.map(v => v.id);
    for (let i = initial.length - 1; i > 0; i--) {const j = Math.floor(Math.random() * (i + 1)); [initial[i], initial[j]] = [initial[j], initial[i]];}
    state.result = {initial, gate:{passed:initial, rejected:[]}, bm25:[], ranked:[]};
    setStage(0); $("#journey-status").textContent = "Получаем результаты анализа…";
    try {
      const result = await api("/api/demo/pipeline", {method:"POST", body:JSON.stringify({resume_id:state.selected})});
      if (request !== state.request) return;
      if (result.resume_id !== state.selected || !Array.isArray(result.initial) || !Array.isArray(result.gate?.passed) || !Array.isArray(result.bm25) || !Array.isArray(result.ranked)) throw new Error("Сервер вернул некорректный результат анализа.");
      state.result = result; state.busy = false; $("#journey-status").textContent = "Результаты получены. Переходите к следующему этапу."; setStage(0);
    } catch (error) { if (request !== state.request) return; state.busy = false; $("#journey-status").textContent = "Не удалось получить результаты. Вернитесь к выбору и попробуйте снова."; $("#error").textContent = error.message; }
  });
  $("#next").addEventListener("click", () => {if (!state.busy && state.stage < 3) setStage(state.stage + 1);});
  $("#back").addEventListener("click", () => {state.request++; state.busy = false; state.stage = -1; $("#journey").hidden = true; $("#choose").hidden = false; $("#error").textContent = ""; $("#section-label").textContent = "01 / ТОЧКА ОТСЧЁТА"; $("#heading").innerHTML = 'Найдём <em>своё</em><br>совпадение<span class="red">.</span>'; $("#subtitle").textContent = "Одно резюме. Все вакансии. И понятный путь между ними."; choice(); explain(0);});
  if (state.token) enter();
})(typeof window === "undefined" ? globalThis : window);
