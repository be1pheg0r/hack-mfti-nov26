"use strict";
const $ = (selector) => document.querySelector(selector);
const state = {token: sessionStorage.getItem("svodka-token") || "", data: null, selected: null, screen: "choose", current: null, pinned: []};
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
  const allowed = {choose: ["loading"], loading: ["choose", "result"], result: ["choose"]};
  if (next !== state.screen && !allowed[state.screen].includes(next)) return;
  state.screen = next;
  for (const name of ["choose", "loading", "result"]) {
    $("#" + name).hidden = name !== next;
    document.querySelector(`[data-step="${name}"]`).classList.toggle("active", name === next);
  }
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
