"use strict";
(function (root, factory) {
  const ui = factory();
  if (typeof module === "object" && module.exports) module.exports = ui;
  if (root) root.GateUI = ui;
})(typeof window === "undefined" ? globalThis : window, function () {
  const FILTERS = [
    ["experience", "Опыт"],
    ["education", "Образование"],
    ["work_format", "Формат работы"],
  ];
  const STATUS = {
    pass: ["Соответствует", "pass"],
    fail: ["Не соответствует", "fail"],
    unknown: ["Недостаточно данных", "unknown"],
  };
  const FILTER_NAMES = Object.fromEntries(FILTERS);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '\"': "&quot;", "'": "&#39;"})[c]);
  const statusOf = (value) => STATUS[value] ? value : "unknown";
  function normalize(vacancy) {
    const decisions = Array.isArray(vacancy.gate?.decisions) ? vacancy.gate.decisions : [];
    const byFilter = new Map(decisions.map((decision) => [decision.filter, decision]));
    const ordered = FILTERS.map(([filter]) => byFilter.get(filter) || {filter, status: "unknown", explanation: "Проверка остановлена на предыдущем правиле.", missing: true});
    for (const decision of decisions) if (!FILTER_NAMES[decision.filter]) ordered.push(decision);
    const missing = FILTERS.some(([filter]) => !byFilter.has(filter));
    const suppliedStatus = statusOf(vacancy.gate?.status);
    const status = suppliedStatus === "pass" && missing ? "fail" : suppliedStatus;
    return {decisions: ordered, status};
  }
  function summarize(payload) {
    const counts = {total: 0, pass: 0, fail: 0, unknown: 0};
    for (const vacancy of (Array.isArray(payload?.vacancies) ? payload.vacancies : [])) { counts[normalize(vacancy).status]++; counts.total++; }
    return counts;
  }
  function safeUrl(value) {
    try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) ? url.href : ""; } catch { return ""; }
  }
  function entityText(entity) {
    if (entity == null) return "";
    if (typeof entity !== "object") return String(entity);
    return entity.display_value ?? entity.value ?? entity.label ?? "";
  }
  function explanationText(decision) {
    const explanation = String(decision.explanation || "");
    const known = {
      "Vacancy experience requirement is missing.": "В вакансии не указано требование к опыту.",
      "Resume experience is not stated; it is unknown, not zero.": "Опыт в резюме не указан. Это неизвестное значение, а не нулевой опыт.",
      "Experience is invalid or not confirmed.": "Опыт указан некорректно или не подтверждён.",
      "Vacancy education requirement is missing.": "В вакансии не указано требование к образованию.",
      "Resume education is not stated; it is unknown.": "Образование в резюме не указано.",
      "No confirmed completed education level; incomplete/unclear evidence remains unknown.": "Не найден подтверждённый завершённый уровень образования; неполные или неоднозначные данные нельзя считать подтверждением.",
      "Vacancy contains an education level not present in the configured ordering.": "Уровень образования из вакансии отсутствует в настроенной шкале сравнения.",
      "Higher education is confirmed, but its degree is unspecified.": "Высшее образование подтверждено, но степень не указана.",
      "Vacancy work format is missing.": "В вакансии не указан формат работы.",
      "Resume explicitly excludes the required work format.": "В резюме явно исключён требуемый формат работы.",
      "Resume work format is not confirmed; it is unknown.": "Формат работы в резюме не подтверждён.",
      "Vacancy work format is not configured.": "Для формата работы из вакансии не настроено правило совместимости.",
      "No candidate format is compatible with the vacancy format.": "Формат работы из резюме несовместим с требованиями вакансии.",
    };
    if (known[explanation]) return known[explanation];
    const experience = explanation.match(/^Resume experience ([\d.+-]+) years minus requirement ([\d.+-]+) years = ([\d.+-]+)\.(.*)$/);
    if (experience) return `Опыт кандидата — ${experience[1]} г.; требование — ${experience[2]} г.; разница — ${experience[3]} г.${experience[4] ? " Суммарные периоды могут пересекаться: в источнике указано, что пересечения или дубли не проверялись." : ""}`;
    const education = explanation.match(/^Highest confirmed completed education rank difference \(resume - vacancy\) is (-?\d+)\.$/);
    if (education) return `Разница уровней завершённого образования (резюме минус вакансия): ${education[1]}.`;
    const compatible = explanation.match(/^Compatible candidate format\(s\): (.+)\.$/);
    if (compatible) return `Совместимый формат работы по данным резюме: ${compatible[1].split(", ").map((format) => ({remote: "удалённый", hybrid: "гибридный", on_site: "офисный"}[format] || format)).join(", ")}.`;
    return explanation || "Пояснение не предоставлено.";
  }
  function evidence(entity) {
    if (!entity || typeof entity !== "object") return "";
    const quote = entity.evidence ?? entity.quote ?? entity.evidence_text;
    const modelQuote = entity.model_quote;
    const context = entity.context;
    return [quote ? `<blockquote><strong>Цитата в источнике:</strong> ${escapeHtml(quote)}</blockquote>` : "", modelQuote && modelQuote !== quote ? `<blockquote><strong>Цитата модели:</strong> ${escapeHtml(modelQuote)}</blockquote>` : "", context ? `<p class="gate-context"><strong>Контекст:</strong> ${escapeHtml(context)}</p>` : ""].join("");
  }
  function entities(label, entries) {
    if (!Array.isArray(entries) || !entries.length) return `<p class="gate-no-data">${label}: данные отсутствуют.</p>`;
    return `<div class="gate-entities"><h4>${label}</h4>${entries.map((entity) => `<div class="gate-entity"><strong>${escapeHtml(entityText(entity) || "Сущность без значения")}</strong>${entity?.label || entity?.importance || entity?.polarity === "absent" ? `<small>${[entity.label, entity.importance, entity.polarity === "absent" ? "Явное отсутствие" : ""].filter(Boolean).map(escapeHtml).join(" · ")}</small>` : ""}${evidence(entity)}</div>`).join("")}</div>`;
  }
  function decisionHtml(decision) {
    const status = statusOf(decision.status);
    const [label, className] = decision.missing ? ["Не проверено", "unknown"] : STATUS[status];
    return `<details class="gate-rule gate-rule-${className}" ${!decision.missing && status !== "pass" ? "open" : ""}><summary><span>${escapeHtml(FILTER_NAMES[decision.filter] || decision.filter || "Правило")}</span><span class="gate-pill gate-pill-${className}">${label}</span></summary><div class="gate-rule-body"><p>${escapeHtml(explanationText(decision))}</p>${decision.difference !== undefined && decision.difference !== null ? `<p><strong>Различие:</strong> ${escapeHtml(decision.difference)}</p>` : ""}${entities("Требование вакансии", decision.requirement)}${entities("Данные резюме", decision.candidate)}</div></details>`;
  }
  function render(payload) {
    const vacancies = Array.isArray(payload?.vacancies) ? payload.vacancies : [];
    const summary = summarize(payload);
    const cards = vacancies.map((vacancy, index) => {
      const result = normalize(vacancy);
      const [label, className] = STATUS[result.status];
      const url = safeUrl(vacancy.source_url);
      return `<article class="gate-vacancy" aria-labelledby="gate-vacancy-${index}"><div class="gate-vacancy-heading"><div><span class="gate-overline">ВАКАНСИЯ ${String(index + 1).padStart(2, "0")}</span><h3 id="gate-vacancy-${index}">${escapeHtml(vacancy.title || "Без названия")}</h3></div><span class="gate-pill gate-pill-${className}">${label}</span></div>${url ? `<a class="gate-source" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">Открыть источник ↗</a>` : `<span class="gate-no-data">Ссылка на источник отсутствует.</span>`}<div class="gate-rules"><h4>Проверка правил</h4>${result.decisions.map(decisionHtml).join("")}</div></article>`;
    }).join("");
    const count = (key, label, className) => `<div class="gate-count gate-count-${className}"><strong>${summary[key]}</strong><span>${label}</span></div>`;
    return `<div class="gate-summary" role="group" aria-label="Итоги проверки"><div class="gate-count"><strong>${summary.total}</strong><span>всего вакансий</span></div>${count("pass", "соответствует", "pass")}${count("fail", "не соответствует", "fail")}${count("unknown", "недостаточно данных", "unknown")}</div>${vacancies.length ? `<div class="gate-vacancies">${cards}</div>` : `<div class="gate-empty"><h3>Вакансий нет</h3><p>Для выбранного резюме результаты Gate 1 не получены.</p></div>`}`;
  }
  return {normalize, render, summarize};
});
// --- Demo application ---
const $ = (selector) => document.querySelector(selector);
const state = {token: sessionStorage.getItem("svodka-token") || "", data: null, selected: null, screen: "choose", current: null, pinned: [], gate: null, gateBusy: false};
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function api(path, options = {}) {
  const bytes = new TextEncoder().encode(state.token);
  const encoded = btoa(Array.from(bytes, (byte) => String.fromCharCode(byte)).join("")).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  const response = await fetch(path, {...options, headers: {"Authorization": `Bearer utf8.${encoded}`, ...(options.body ? {"Content-Type": "application/json"} : {})}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Ошибка соединения");
  return data;
}
function transition(next) {
  const allowed = {choose: ["loading", "gate-loading"], loading: ["choose", "result"], result: ["choose"], "gate-loading": ["choose", "gate-result"], "gate-result": ["choose", "gate-loading"]};
  if (next !== state.screen && !allowed[state.screen].includes(next)) return;
  state.screen = next;
  const screens = ["choose", "loading", "result", "gate-loading", "gate-result"];
  for (const name of screens) $("#" + name).hidden = name !== next;
  const step = next.startsWith("gate-") ? (next === "gate-loading" ? "loading" : "result") : next;
  document.querySelectorAll("[data-step]").forEach((item) => item.classList.toggle("active", item.dataset.step === step));
}
async function enter() {
  try {
    state.data = await api("/api/demo");
    sessionStorage.setItem("svodka-token", state.token);
    state.selected = state.data.resumes[0]?.id;
    if (!state.selected) throw new Error("Резюме не найдены.");
    $("#login").hidden = true; $("#app").hidden = false;
    $("#steps").hidden = false; $("#logout").hidden = false; $("#vacancies-button").hidden = false;
    $(".footnote").textContent = "Текст DOCX: без изображений и колонтитулов · внешняя LLM · рейтинг вакансий — макет";
    transition("choose"); renderChoice();
  } catch (error) { $("#login-error").textContent = error.message; }
}
function renderChoice() {
  const resumes = state.data.resumes;
  $("#resume-count").textContent = `${String(resumes.length).padStart(2, "0")} РЕЗЮМЕ`;
  $("#resume-list").innerHTML = resumes.map((r, i) => `<button type="button" class="resume-choice" data-id="${escapeHtml(r.id)}" aria-pressed="${r.id === state.selected}"><span class="choice-number">${String(i + 1).padStart(2, "0")}</span><span><strong>${escapeHtml(r.name)}</strong><small>${escapeHtml(r.title)}</small></span><span class="choice-arrow">↗</span></button>`).join("");
  const index = resumes.findIndex((r) => r.id === state.selected);
  const person = resumes[index];
  $("#preview-number").textContent = String(index + 1).padStart(2, "0");
  $("#preview-name").textContent = person.name;
  $("#preview-title").textContent = person.title;
  $("#preview-text").textContent = person.text;
}
function showVacancy(id) {
  const vacancy = state.data.vacancies.find((item) => item.id === id);
  if (!vacancy) return;
  state.pinned = [];
  $("#vacancy-list").innerHTML = state.data.vacancies.map((item) => `<button type="button" class="vacancy-choice" data-vacancy="${escapeHtml(item.id)}" aria-pressed="${item.id === id}">${escapeHtml(item.title)}</button>`).join("");
  $("#vacancy-title").textContent = vacancy.title;
  $("#vacancy-text").textContent = vacancy.text;
  const entities = vacancy.entities ?? [];
  $("#vacancy-entity-count").textContent = String(entities.length).padStart(2, "0");
  $("#vacancy-entities").innerHTML = entities.length ? entities.map((e, i) => `<button type="button" class="entity" data-ids="${i}"><small>${escapeHtml(e.label)}${e.importance ? ` · ${escapeHtml(e.importance)}` : ""}</small><strong>${escapeHtml(e.display_value ?? e.value)}</strong>${e.context ? `<span class="entity-context">${escapeHtml(e.context)}</span>` : ""}</button>`).join("") : "<p class=\"findings-hint\">Сущности вакансии ещё не сохранены.</p>";
  renderSource(vacancy, "#vacancy-text", "фрагмент вакансии");
  if (!$("#vacancy-dialog").open) $("#vacancy-dialog").showModal();
}
function renderSource(person, selector = "#result-text", kind = "резюме") {
  const chars = Array.from(person.text);
  const valid = (person.entities ?? []).flatMap((entity, index) => (entity.sources ?? [entity]).map((source) => ({...source, index}))).filter((e) => Number.isInteger(e.start) && Number.isInteger(e.end) && e.start >= 0 && e.end <= chars.length && e.start < e.end && chars.slice(e.start, e.end).join("") === e.evidence);
  const edges = [...new Set([0, chars.length, ...valid.flatMap((e) => [e.start, e.end])])].sort((a, b) => a - b);
  const node = $(selector); node.replaceChildren();
  for (let i = 0; i < edges.length - 1; i++) {
    const start = edges[i], end = edges[i + 1];
    const ids = valid.filter((e) => e.start <= start && e.end >= end).map((e) => e.index);
    const text = chars.slice(start, end).join("");
    if (!ids.length) { node.append(document.createTextNode(text)); continue; }
    const mark = document.createElement("span");
    mark.className = "source-span"; mark.dataset.ids = ids.join(",");
    mark.setAttribute("tabindex", "0"); mark.setAttribute("role", "button");
    mark.setAttribute("aria-label", `${kind}: ${text}`); mark.textContent = text;
    node.append(mark);
  }
}
function highlight(root, ids) {
  root.querySelectorAll("[data-ids]").forEach((element) => {
    const active = element.dataset.ids.split(",").some((id) => ids.includes(id));
    element.classList.toggle("is-active", active);
  });
}
function renderResult(person) {
  state.current = person; state.pinned = [];
  $("#result-name").textContent = person.name;
  $("#result-position").textContent = person.title;
  $("#entity-count").textContent = String(person.entities.length).padStart(2, "0");
  $("#entities").innerHTML = person.entities.length ? person.entities.map((e, i) => `<button type="button" class="entity" data-ids="${i}"><small>${escapeHtml(e.label)}${e.polarity === "absent" ? " · Явное отрицание" : ""}</small><strong>${escapeHtml(e.display_value ?? e.value)}</strong>${e.context ? `<span class="entity-context">${escapeHtml(e.context)}</span>` : ""}${e.evidence_kind === "source_block" ? "<span class=\"entity-context\">Основание — абзац: цитата модели отличается типографикой.</span>" : ""}</button>`).join("") : "<p>Модель не вернула сущностей.</p>";
  $("#rejected").textContent = person.rejected ? `Отклонено при проверке: ${person.rejected_quotes} без точной цитаты, ${person.rejected_values} с недопустимым значением.` : "";
  const order = person.id === "r3" ? ["accountant", "frontend", "curator"] : ["frontend", "curator", "accountant"];
  $("#jobs").innerHTML = order.map((id, i) => { const v = state.data.vacancies.find((item) => item.id === id); return `<button type="button" class="job vacancy-choice" data-vacancy="${escapeHtml(id)}"><b>0${i + 1}</b>${escapeHtml(v.title)} ↗</button>`; }).join("");
  renderSource(person);
}
$("#login-form").addEventListener("submit", (event) => { event.preventDefault(); state.token = $("#token").value; enter(); });
$("#logout").addEventListener("click", () => { sessionStorage.removeItem("svodka-token"); location.reload(); });
$("#vacancies-button").addEventListener("click", () => showVacancy(state.data.vacancies[0].id));
$("#vacancy-close").addEventListener("click", () => { state.pinned = []; $("#vacancy-dialog").close(); });
$("#vacancy-list").addEventListener("click", (event) => { const button = event.target.closest("[data-vacancy]"); if (button) showVacancy(button.dataset.vacancy); });
$("#jobs").addEventListener("click", (event) => { const button = event.target.closest("[data-vacancy]"); if (button) showVacancy(button.dataset.vacancy); });
$("#resume-list").addEventListener("click", (event) => { const button = event.target.closest("button[data-id]"); if (!button) return; state.selected = button.dataset.id; renderChoice(); });
$("#back").addEventListener("click", () => { transition("choose"); renderChoice(); });
async function runGate() {
  if (state.gateBusy || !state.selected) return;
  state.gateBusy = true;
  $("#run-gate").disabled = true;
  $("#gate-retry").disabled = true;
  $("#choose-error").textContent = "";
  $("#gate-error").hidden = true;
  $("#gate-loading-status").textContent = "Получаем результаты для всех вакансий…";
  transition("gate-loading");
  try {
    const result = await api("/api/demo/gate", {method: "POST", body: JSON.stringify({resume_id: state.selected})});
    if (result.resume_id !== state.selected || !Array.isArray(result.vacancies)) throw new Error("Сервер вернул некорректные результаты Gate 1.");
    state.gate = result;
    const person = state.data.resumes.find((resume) => resume.id === state.selected);
    $("#gate-resume-label").textContent = person ? `Резюме: ${person.name} · ${person.title}` : `ID резюме: ${state.selected}`;
    $("#gate-results").innerHTML = window.GateUI.render(result);
    transition("gate-result");
    $("#gate-result-title").focus?.({preventScroll: true});
  } catch (error) {
    transition("choose");
    $("#choose-error").textContent = error.message || "Не удалось загрузить результаты Gate 1.";
    $("#gate-error").textContent = error.message || "Не удалось загрузить результаты Gate 1.";
  } finally {
    state.gateBusy = false;
    $("#run-gate").disabled = false;
    $("#gate-retry").disabled = false;
  }
}
$("#run-gate").addEventListener("click", runGate);
$("#gate-retry").addEventListener("click", runGate);
$("#gate-back").addEventListener("click", () => { $("#gate-error").hidden = true; transition("choose"); renderChoice(); });
$("#analyze").addEventListener("click", async () => {
  if (state.screen !== "choose") return;
  $("#choose-error").textContent = "";
  transition("loading");
  try {
    const [result] = await Promise.all([api("/api/demo/analyze", {method: "POST", body: JSON.stringify({vacancy_id: state.data.vacancies[0].id, resume_id: state.selected})}), delay(matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 1000)]);
    const person = result.resumes.find((r) => r.id === state.selected);
    if (!person) throw new Error("Результат выбранного резюме не найден.");
    renderResult(person); transition("result");
  } catch (error) { transition("choose"); $("#choose-error").textContent = error.message; }
});
function revealCounterpart(root, item) {
  const ids = item.dataset.ids.split(",");
  const entity = item.classList.contains("entity");
  const selector = entity ? ".source-span" : ".entity";
  const target = [...root.querySelectorAll(selector)].find((node) => node.dataset.ids.split(",").some((id) => ids.includes(id)));
  if (!target) return;
  const scroller = target.closest(entity ? (root.id === "vacancy-dialog" ? ".vacancy-source" : ".document-body") : (root.id === "vacancy-dialog" ? ".vacancy-findings" : ".findings-body"));
  const parent = scroller.getBoundingClientRect(), rect = target.getBoundingClientRect();
  if (rect.top < parent.top || rect.bottom > parent.bottom) scroller.scrollTop += rect.top - parent.top - 20;
}
function connectHighlights(root) {
  root.addEventListener("mouseover", (event) => { const item = event.target.closest("[data-ids]"); if (item && root.contains(item)) { highlight(root, item.dataset.ids.split(",")); revealCounterpart(root, item); } });
  root.addEventListener("mouseout", (event) => { if (event.target.closest("[data-ids]") && !event.relatedTarget?.closest?.("[data-ids]")) highlight(root, state.pinned); });
  root.addEventListener("focusin", (event) => { const item = event.target.closest("[data-ids]"); if (item) { highlight(root, item.dataset.ids.split(",")); revealCounterpart(root, item); } });
  root.addEventListener("focusout", () => highlight(root, state.pinned));
  root.addEventListener("click", (event) => { const item = event.target.closest("[data-ids]"); if (item) { state.pinned = item.dataset.ids.split(","); highlight(root, state.pinned); revealCounterpart(root, item); } });
  root.addEventListener("keydown", (event) => { const item = event.target.closest(".source-span"); if (item && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); state.pinned = item.dataset.ids.split(","); highlight(root, state.pinned); revealCounterpart(root, item); } });
}
connectHighlights($("#result"));
connectHighlights($("#vacancy-dialog"));
if (state.token) enter();
