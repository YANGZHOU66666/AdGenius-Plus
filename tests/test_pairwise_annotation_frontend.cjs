/* Offline UI regressions. Run with Node; no browser, dependencies or file writes. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.dataset = {};
    this.style = {};
    this.textContent = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.children = [];
    this.attributes = {};
    this.events = new Map();
    const classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => classes.add(name)),
      remove: (...names) => names.forEach((name) => classes.delete(name)),
      toggle: (name, force) => {
        const selected = force === undefined ? !classes.has(name) : force;
        if (selected) classes.add(name); else classes.delete(name);
        return selected;
      },
      contains: (name) => classes.has(name),
    };
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(type, callback) { this.events.set(type, callback); }
  append(...children) {
    children.forEach((child) => this.children.push(
      ...(child.fragment ? child.children : [child])));
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  querySelectorAll(tag) {
    return this.children.flatMap((child) => [
      ...(child.tagName === tag.toUpperCase() ? [child] : []),
      ...child.querySelectorAll(tag),
    ]);
  }
  querySelector(tag) {
    const existing = this.querySelectorAll(tag)[0];
    if (existing) return existing;
    const element = new Element(tag);
    this.children.push(element);
    return element;
  }
  click() { return this.events.get("click")?.({ preventDefault() {} }); }
}

function clone(value) { return JSON.parse(JSON.stringify(value)); }
function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}
function response(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status,
    json: async () => clone(payload) };
}
function state(annotator = "zhou", annotations = {}) {
  return {
    annotator_id: annotator,
    annotation_set_id: "gold_pairwise_v1",
    protocol_version: "pairwise-v0.1",
    dataset_revision: "dataset_1",
    annotations_revision: Object.keys(annotations).length ? "labels_1" : "labels_0",
    drafts_revision: "drafts_0",
    pairs: [{ pair_id: "pair_0", product_id: "gold_product_0",
      product_title: "A metal stapler", product_information: "Metal body; 20 sheets.",
      candidates: [{ candidate_id: "candidate_first", text: "A metal stapler." },
        { candidate_id: "candidate_second", text: "Staple up to 20 sheets." }] }],
    annotations: clone(annotations), drafts: {}, paths: {},
  };
}
function annotation(annotator = "zhou", winner = "candidate_first") {
  return {
    annotation_id: "human_1", judge_type: "human",
    annotation_set_id: "gold_pairwise_v1", pair_id: "pair_0",
    annotator_id: annotator, protocol_version: "pairwise-v0.1",
    presentation: { A: "candidate_first", B: "candidate_second" },
    pairwise_result: { status: "success", choice: "A",
      winner_candidate_id: winner, reason: null, error: null },
  };
}

function harness() {
  const elements = new Map();
  const intervals = new Map();
  let timerId = 0;
  let randomByte = 0;
  let fetchImpl = async (url) => url === "/api/config"
    ? response({ default_annotator: "zhou" }) : response(state());
  const window = {
    location: { search: "?annotator_id=zhou", href: "http://127.0.0.1:8765/?annotator_id=zhou" },
    crypto: { getRandomValues: (values) => values.fill(randomByte) },
    addEventListener() {}, scrollTo() {},
    history: { replaceState() {} },
  };
  const context = vm.createContext({
    window, document: {
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
      },
      createElement: (tag) => new Element(tag),
      createDocumentFragment: () => Object.assign(new Element(), { fragment: true }),
      addEventListener() {},
    },
    URLSearchParams, URL, Uint8Array, Blob, console,
    HTMLTextAreaElement: class {}, HTMLInputElement: class {},
    fetch: (...args) => fetchImpl(...args),
    setTimeout: () => ++timerId, clearTimeout() {},
    setInterval: (callback) => { const id = ++timerId; intervals.set(id, callback); return id; },
    clearInterval: (id) => intervals.delete(id),
  });
  const sourcePath = path.join(__dirname, "../tools/pairwise_annotation/app.js");
  let source = fs.readFileSync(sourcePath, "utf8");
  const ending = /\n  load\(\{ shuffle: true \}\);\n\}\)\(\);\s*$/;
  assert.match(source, ending, "UI bootstrap remains identifiable for isolated offline tests");
  source = source.replace(ending,
    "\n  window.testAPI = {app, load, syncState, choose, saveSnapshot, switchAnnotator, navigateTo};\n})();");
  vm.runInContext(source, context, { filename: sourcePath });
  return {
    api: window.testAPI,
    element: (id) => elements.get(id),
    setFetch: (callback) => { fetchImpl = callback; },
    setRandomByte: (value) => { randomByte = value; },
  };
}

test("a poll begun before save cannot replace the newer saved annotation", async () => {
  const ui = harness();
  await ui.api.load();
  const oldPoll = deferred();
  const saved = annotation();
  ui.setFetch(async (url, options = {}) => {
    if (options.method === "PUT") {
      return response({ annotation: saved, annotations_revision: "labels_1" });
    }
    return oldPoll.promise;
  });
  const poll = ui.api.syncState();
  ui.api.choose("A");
  await ui.api.app.saveQueue;
  assert.equal(ui.element("done-count").textContent, "1");
  oldPoll.resolve(response(state()));
  await poll;
  assert.equal(ui.api.app.annotations.pair_0.annotation_id, saved.annotation_id);
  assert.equal(ui.element("done-count").textContent, "1");
  assert.equal(ui.element("choose-a").getAttribute("aria-pressed"), "true");
});

test("a response for the previous annotator cannot mix labels after switching", async () => {
  const ui = harness();
  await ui.api.load();
  const oldPoll = deferred();
  ui.setFetch(async (url) => {
    if (url.includes("annotator_id=zhou")) return oldPoll.promise;
    return response(state("chengkai"));
  });
  const poll = ui.api.syncState();
  ui.element("annotator-input").value = "chengkai";
  await ui.api.switchAnnotator({ preventDefault() {} });
  oldPoll.resolve(response(state("zhou", { pair_0: annotation("zhou") })));
  await poll;
  assert.equal(ui.api.app.annotatorId, "chengkai");
  assert.equal(ui.api.app.state.annotator_id, "chengkai");
  assert.equal(Object.keys(ui.api.app.annotations).length, 0);
  assert.equal(ui.element("done-count").textContent, "0");
});

test("failed saves stay visibly failed after finalization and healthy polling", async () => {
  const ui = harness();
  await ui.api.load();
  ui.setFetch(async (_url, options = {}) => options.method === "PUT"
    ? response({ error: "Disk write failed" }, 500) : response(state()));
  ui.api.choose("A");
  await ui.api.app.saveQueue;
  assert.equal(ui.element("sync-status").dataset.state, "error");
  assert.equal(ui.api.app.dirty, true);
  assert.match(ui.element("notice-message").textContent, /Disk write failed/);
  await ui.api.syncState();
  assert.equal(ui.element("sync-status").dataset.state, "error");
  assert.equal(ui.api.app.dirty, true);
  assert.match(ui.element("notice-message").textContent, /Disk write failed/);
});

test("revisiting reshuffles sides while the saved winner follows the actual candidate", async () => {
  const ui = harness();
  let snapshot = state();
  snapshot.pairs.push({ pair_id: "pair_1", product_id: "gold_product_1",
    product_title: "Second product", product_information: "Another supported product.",
    candidates: [{ candidate_id: "second_first", text: "First ad for second product." },
      { candidate_id: "second_second", text: "Second ad for second product." }] });
  let writes = 0;
  ui.setFetch(async (_url, options = {}) => {
    if (options.method === "PUT") {
      writes += 1;
      const body = JSON.parse(options.body);
      const saved = annotation();
      saved.presentation = body.presentation;
      saved.pairwise_result.choice = body.choice;
      saved.pairwise_result.winner_candidate_id = body.presentation[body.choice];
      snapshot = { ...snapshot, annotations_revision: "labels_1",
        annotations: { pair_0: saved } };
      return response({ annotation: saved, annotations_revision: "labels_1" });
    }
    return response(snapshot);
  });
  await ui.api.load();
  assert.equal(ui.element("copy-a").textContent, "A metal stapler.");
  ui.api.choose("A");
  await ui.api.app.saveQueue;
  await ui.api.navigateTo(1);
  ui.setRandomByte(255);
  await ui.api.navigateTo(0);
  assert.equal(ui.element("copy-a").textContent, "Staple up to 20 sheets.");
  assert.equal(ui.element("copy-b").textContent, "A metal stapler.");
  assert.equal(ui.element("choose-a").getAttribute("aria-pressed"), "false");
  assert.equal(ui.element("choose-b").getAttribute("aria-pressed"), "true");
  assert.equal(ui.api.app.annotations.pair_0.pairwise_result.winner_candidate_id,
    "candidate_first");
  assert.equal(writes, 1, "browsing and reshuffling must not rewrite labels");
});
