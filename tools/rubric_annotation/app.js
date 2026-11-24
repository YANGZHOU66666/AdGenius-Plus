"use strict";

const LABELS = {
  factual_support: {name: "factual_support · Does the ad conflict with the information?",
                    ok: ["No conflict", "True"], bad: ["Conflict", "False"],
                    placeholder: "If False: the ad sentence ↔ the conflicting information text"},
  concrete_product_info: {name: "concrete_product_info · Does the ad correctly convey specific product information?",
                          ok: ["Has concrete info", "True"], bad: ["No concrete info", "False"],
                          placeholder: "The specific information conveyed and where it comes from"},
};

const $ = (id) => document.getElementById(id);
const S = {data: null, index: 0, pending: {}, queue: Promise.resolve(), timer: null};
const annotator = new URLSearchParams(location.search).get("annotator_id") || "zhou";

function item() { return S.data.items[S.index]; }
function saved(cid) { return S.data.annotations[cid] || null; }
function savedValues(record) {
  const out = {values: {}, reasons: {}};
  if (record) for (const c of record.rubric_result.criteria) { out.values[c.criterion_id] = c.value; out.reasons[c.criterion_id] = c.reason || ""; }
  return out;
}
function current(cid) {
  if (!S.pending[cid]) S.pending[cid] = savedValues(saved(cid));
  return S.pending[cid];
}
function complete(entry) { return S.data.criteria.every((c) => typeof entry.values[c] === "boolean"); }
function isFalse(record) { return record && record.rubric_result.criteria.some((c) => c.value === false); }

function setSync(state, text) { const el = $("sync"); el.dataset.state = state; el.textContent = text; }
function toast(text) { const el = $("toast"); el.textContent = text; el.hidden = false; clearTimeout(el._t); el._t = setTimeout(() => { el.hidden = true; }, 1600); }
function fail(message) { $("error-text").textContent = message; $("error").hidden = false; setSync("error", "Not saved"); }

async function api(method, path, body) {
  const response = await fetch(path, {method, headers: body ? {"Content-Type": "application/json"} : {}, body: body ? JSON.stringify(body) : undefined});
  const data = await response.json().catch(() => ({error: `HTTP ${response.status}`}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function load() {
  try {
    S.data = await api("GET", `/api/state?annotator_id=${encodeURIComponent(annotator)}`);
  } catch (error) { $("loading").textContent = `Could not load: ${error.message}`; return; }
  const fromHash = S.data.items.findIndex((it) => it.item_id === location.hash.slice(1));
  const firstOpen = S.data.items.findIndex((it) => !saved(it.candidate_id));
  S.index = fromHash >= 0 ? fromHash : Math.max(firstOpen, 0);
  $("loading").hidden = true; $("workspace").hidden = false;
  $("file-path").textContent = `Labels: ${S.data.paths.annotations}`;
  setSync("ok", "Synced with local file");
  buildCriteria(); render();
}

function buildCriteria() {
  const box = $("criteria"); box.textContent = "";
  for (const c of S.data.criteria) {
    const L = LABELS[c];
    const wrap = document.createElement("div"); wrap.className = "criterion";
    wrap.innerHTML = `<div class="criterion-name"></div><div class="choices">
      <button type="button" class="choice ok" data-value="true" aria-pressed="false"><span></span><small></small></button>
      <button type="button" class="choice bad" data-value="false" aria-pressed="false"><span></span><small></small></button></div>
      <textarea class="reason" rows="2"></textarea>`;
    wrap.querySelector(".criterion-name").textContent = L.name;
    const [ok, bad] = wrap.querySelectorAll(".choice");
    ok.querySelector("span").textContent = L.ok[0]; ok.querySelector("small").textContent = L.ok[1];
    bad.querySelector("span").textContent = L.bad[0]; bad.querySelector("small").textContent = L.bad[1];
    ok.onclick = () => choose(c, true); bad.onclick = () => choose(c, false);
    const reason = wrap.querySelector("textarea"); reason.placeholder = L.placeholder; reason.dataset.criterion = c;
    reason.oninput = () => { current(item().candidate_id).reasons[c] = reason.value; scheduleReasonSave(); };
    reason.onblur = () => flushReasonSave();
    wrap.dataset.criterion = c; box.appendChild(wrap);
  }
}

function render() {
  const it = item(), record = saved(it.candidate_id), entry = current(it.candidate_id);
  const total = S.data.items.length, done = Object.keys(S.data.annotations).length;
  $("done-count").textContent = done; $("total-count").textContent = total;
  $("false-count").textContent = Object.values(S.data.annotations).filter(isFalse).length;
  $("progress-bar").style.width = `${(100 * done / total).toFixed(1)}%`;
  $("item-number").textContent = `ITEM ${String(S.index + 1).padStart(3, "0")}`;
  $("page").textContent = `${S.index + 1} / ${total}`;
  const status = $("item-status");
  status.dataset.value = record ? String(!isFalse(record)) : "";
  status.textContent = record ? (isFalse(record) ? "Labeled · False" : "Labeled · True") : (Object.keys(entry.values).length ? "Partly chosen, not saved" : "Unlabeled");
  $("ad").textContent = it.ad;
  $("product-title").textContent = it.product_title;
  $("product-information").textContent = it.product_information;
  for (const wrap of $("criteria").children) {
    const c = wrap.dataset.criterion;
    for (const b of wrap.querySelectorAll(".choice")) b.setAttribute("aria-pressed", String(entry.values[c] === (b.dataset.value === "true")));
    const reason = wrap.querySelector("textarea");
    if (document.activeElement !== reason) reason.value = entry.reasons[c] || "";
  }
  $("clear").disabled = !record; $("prev").disabled = S.index === 0; $("next").disabled = S.index === total - 1;
  renderGrid();
  if (location.hash.slice(1) !== it.item_id) history.replaceState(null, "", `${location.search}#${it.item_id}`);
}

function renderGrid() {
  const grid = $("grid");
  if (grid.children.length !== S.data.items.length) {
    grid.textContent = "";
    S.data.items.forEach((it, i) => { const b = document.createElement("button"); b.type = "button"; b.textContent = i + 1; b.onclick = () => go(i); grid.appendChild(b); });
  }
  S.data.items.forEach((it, i) => {
    const r = saved(it.candidate_id), b = grid.children[i];
    b.className = (r ? (isFalse(r) ? "bad" : "ok") : "") + (i === S.index ? " current" : "");
    b.title = r ? (isFalse(r) ? "False" : "True") : "Unlabeled";
  });
}

function go(index) {
  flushReasonSave();
  S.index = Math.max(0, Math.min(S.data.items.length - 1, index));
  render(); window.scrollTo({top: 0}); $("product-information").scrollTop = 0;
}
function nextUnlabeled() {
  const n = S.data.items.length;
  for (let k = 1; k <= n; k++) { const i = (S.index + k) % n; if (!saved(S.data.items[i].candidate_id)) return go(i); }
  toast("All items are labeled");
}

function enqueue(task) { S.queue = S.queue.then(task).catch((error) => fail(error.message)); return S.queue; }

function save(cid) {
  return enqueue(async () => {
    const entry = current(cid);
    if (!complete(entry)) return;
    setSync("saving", "Saving…");
    const result = await api("PUT", "/api/annotation", {annotator_id: annotator, dataset_revision: S.data.dataset_revision,
      candidate_id: cid, expected_annotation: saved(cid), values: entry.values, reasons: entry.reasons});
    S.data.annotations[cid] = result.annotation;
    setSync("ok", "Saved to local file"); render();
  });
}

function choose(criterion, value) {
  const it = item(), entry = current(it.candidate_id);
  entry.values[criterion] = value; render();
  if (!complete(entry)) return;
  save(it.candidate_id).then(() => {
    if (!saved(it.candidate_id)) return;  // The save failed; stay here.
    // Auto-advance only when nothing needs explaining: every criterion true.
    if (S.data.criteria.every((c) => entry.values[c]) && item() === it) setTimeout(() => { if (item() === it) nextUnlabeled(); }, 250);
    else if (!entry.values.factual_support) document.querySelector('textarea[data-criterion="factual_support"]')?.focus();
  });
}

function scheduleReasonSave() {
  clearTimeout(S.timer);
  const cid = item().candidate_id;
  if (saved(cid)) S.timer = setTimeout(() => { S.timer = null; save(cid); }, 700);
}
function flushReasonSave() {
  if (S.timer) { clearTimeout(S.timer); S.timer = null; save(item().candidate_id); }
}

async function clearLabel() {
  const cid = item().candidate_id;
  if (!saved(cid)) return;
  await enqueue(async () => {
    setSync("saving", "Saving…");
    await api("DELETE", "/api/annotation", {annotator_id: annotator, dataset_revision: S.data.dataset_revision,
      candidate_id: cid, expected_annotation: saved(cid)});
    delete S.data.annotations[cid]; delete S.pending[cid];
    setSync("ok", "Label cleared"); render();
  });
}

function highlight() {
  const pre = $("product-information"), text = item().product_information;
  const selected = String(window.getSelection() || "").trim();
  pre.textContent = text;
  if (selected.length < 3 || !$("ad").contains(window.getSelection().anchorNode)) return;
  const pattern = new RegExp(selected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+"), "gi");
  const fragment = document.createDocumentFragment(); let last = 0, first = null, count = 0;
  for (const m of text.matchAll(pattern)) {
    fragment.append(text.slice(last, m.index));
    const mark = document.createElement("mark"); mark.textContent = m[0]; fragment.append(mark);
    first = first || mark; last = m.index + m[0].length; count++;
  }
  fragment.append(text.slice(last)); pre.textContent = ""; pre.append(fragment);
  if (first) first.scrollIntoView({block: "center"}); else toast(`"${selected}" is not in the information`);
  if (count) toast(`${count} match${count > 1 ? "es" : ""} in the information`);
}

document.addEventListener("keydown", (event) => {
  if (!S.data || (event.target instanceof Element && event.target.closest("textarea, input")) || event.metaKey || event.ctrlKey || event.altKey) return;
  const [first] = S.data.criteria;
  if (event.key === "1") choose(first, true);
  else if (event.key === "2") choose(first, false);
  else if (event.key === "ArrowLeft") go(S.index - 1);
  else if (event.key === "ArrowRight") go(S.index + 1);
  else if (event.key.toLowerCase() === "n") nextUnlabeled();
  else return;
  event.preventDefault();
});

$("prev").onclick = () => go(S.index - 1);
$("next").onclick = () => go(S.index + 1);
$("next-unlabeled").onclick = nextUnlabeled;
$("clear").onclick = clearLabel;
$("reload").onclick = () => location.reload();
$("ad").addEventListener("mouseup", () => setTimeout(highlight, 0));
window.addEventListener("beforeunload", flushReasonSave);
load();
