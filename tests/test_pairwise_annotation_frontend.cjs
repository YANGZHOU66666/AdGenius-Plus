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

function withPairs(count, annotations = {}) {
  const snapshot = state("zhou", annotations);
  for (let index = 1; index < count; index += 1) {
    snapshot.pairs.push({ pair_id: `pair_${index}`, product_id: `gold_product_${index}`,
      product_title: `Product ${index}`, product_information: `Supported facts ${index}`,
      candidates: [{ candidate_id: `candidate_${index}_first`, text: `First ad ${index}` },
        { candidate_id: `candidate_${index}_second`, text: `Second ad ${index}` }] });
  }
  return snapshot;
}

function savedFrom(body) {
  const saved = annotation(body.annotator_id, body.presentation[body.choice]);
  saved.pair_id = body.pair_id;
  saved.presentation = body.presentation;
  saved.pairwise_result.choice = body.choice;
  saved.pairwise_result.reason = body.reason;
  return saved;
}

function harness() {
  const elements = new Map();
  const intervals = new Map();
  const documentEvents = new Map();
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
      addEventListener: (type, callback) => documentEvents.set(type, callback),
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
    keydown: (key) => documentEvents.get("keydown")({ key, target: {} }),
  };
}

test("tie choices and the former T shortcut cannot create annotations", async () => {
  const ui = harness();
  await ui.api.load();
  let writes = 0;
  ui.setFetch(async (_url, options = {}) => {
    if (options.method === "PUT") writes += 1;
    return response(state());
  });
  ui.api.choose("tie");
  ui.keydown("t");
  ui.keydown("T");
  await ui.api.app.saveQueue;
  assert.equal(writes, 0);
  assert.equal(ui.api.app.choice, null);
  assert.equal(ui.api.app.dirty, false);
  assert.equal(ui.element("choose-a").getAttribute("aria-pressed"), "false");
  assert.equal(ui.element("choose-b").getAttribute("aria-pressed"), "false");
});

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

test("choosing a winner advances only after the local save succeeds", async () => {
  const ui = harness();
  const snapshot = withPairs(2);
  const pendingSave = deferred();
  const saveStarted = deferred();
  let body;
  ui.setFetch(async (_url, options = {}) => {
    if (options.method === "PUT") {
      body = JSON.parse(options.body);
      saveStarted.resolve();
      return pendingSave.promise;
    }
    return response(snapshot);
  });
  await ui.api.load();
  ui.api.choose("A");
  await saveStarted.promise;
  assert.equal(ui.api.app.index, 0, "an in-flight save must not change the current pair");
  assert.equal(ui.api.app.writePending, 1);
  pendingSave.resolve(response({ annotation: savedFrom(body), annotations_revision: "labels_1" }));
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 1);
  assert.equal(ui.element("question-number").textContent, "PAIR 002");
  assert.equal(ui.api.app.dirty, false);
  assert.equal(ui.api.app.writePending, 0);
  assert.equal(ui.api.app.choice, null);
  assert.equal(ui.api.app.advanceAfterSavePairId, null);
  assert.equal(ui.api.app.annotations.pair_0.pairwise_result.winner_candidate_id,
    "candidate_first");
});

test("automatic navigation skips completed pairs and wraps to an earlier gap", async () => {
  const ui = harness();
  const snapshot = withPairs(5, {
    pair_1: { ...annotation(), pair_id: "pair_1" },
    pair_3: { ...annotation(), pair_id: "pair_3" },
  });
  ui.setFetch(async (_url, options = {}) => options.method === "PUT"
    ? response({ annotation: savedFrom(JSON.parse(options.body)), annotations_revision: "labels_new" })
    : response(snapshot));
  await ui.api.load();
  ui.api.choose("A");
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 2, "pair 1 is already labelled");
  await ui.api.navigateTo(4);
  ui.api.choose("B");
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 2, "after the last pair, wrap past labelled pairs to pair 2");
});

test("failed annotation saves keep the pair visible and retry advances after success", async () => {
  const ui = harness();
  const snapshot = withPairs(2);
  let fail = true;
  ui.setFetch(async (_url, options = {}) => {
    if (options.method !== "PUT") return response(snapshot);
    return fail ? response({ error: "Disk write failed" }, 500)
      : response({ annotation: savedFrom(JSON.parse(options.body)), annotations_revision: "labels_1" });
  });
  await ui.api.load();
  ui.api.choose("A");
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 0);
  assert.equal(ui.api.app.dirty, true);
  assert.equal(ui.api.app.saveError.operation, "save");
  fail = false;
  ui.element("retry-save").click();
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 1);
  assert.equal(ui.api.app.saveError, null);
});

test("saving a reason draft does not count as finishing or move to another pair", async () => {
  const ui = harness();
  const snapshot = withPairs(2);
  let writes = 0;
  ui.setFetch(async (url, options = {}) => {
    if (options.method !== "PUT") return response(snapshot);
    writes += 1;
    assert.equal(url, "/api/draft");
    const body = JSON.parse(options.body);
    return response({ draft: { pair_id: body.pair_id, reason: body.reason },
      drafts_revision: "drafts_1" });
  });
  await ui.api.load();
  ui.element("reason").value = "Need to check the product facts.";
  ui.element("reason").events.get("input")();
  await ui.api.saveSnapshot();
  assert.equal(writes, 1);
  assert.equal(ui.api.app.index, 0);
  assert.equal(ui.api.app.choice, null);
  assert.equal(ui.element("done-count").textContent, "0");
  assert.equal(ui.api.app.advanceAfterSavePairId, null);
});

test("editing the reason of an existing annotation saves without automatically leaving it", async () => {
  const ui = harness();
  const snapshot = withPairs(2, { pair_0: annotation() });
  ui.setFetch(async (_url, options = {}) => options.method === "PUT"
    ? response({ annotation: savedFrom(JSON.parse(options.body)), annotations_revision: "labels_2" })
    : response(snapshot));
  await ui.api.load();
  ui.element("reason").value = "A gives a supported product fact.";
  ui.element("reason").events.get("input")();
  await ui.api.saveSnapshot();
  assert.equal(ui.api.app.index, 0);
  assert.equal(ui.api.app.annotations.pair_0.pairwise_result.reason,
    "A gives a supported product fact.");
  assert.equal(ui.api.app.advanceAfterSavePairId, null);
});

test("choice and reason edits made during a save finish saving before advancing", async () => {
  const ui = harness();
  const snapshot = withPairs(2);
  const firstSave = deferred();
  const secondSave = deferred();
  const firstStarted = deferred();
  const secondStarted = deferred();
  const bodies = [];
  ui.setFetch(async (_url, options = {}) => {
    if (options.method !== "PUT") return response(snapshot);
    bodies.push(JSON.parse(options.body));
    if (bodies.length === 1) {
      firstStarted.resolve();
      return firstSave.promise;
    }
    secondStarted.resolve();
    return secondSave.promise;
  });
  await ui.api.load();
  ui.api.choose("A");
  await firstStarted.promise;
  ui.element("reason").value = "B is clearer and grounded.";
  ui.element("reason").events.get("input")();
  ui.api.choose("B");
  firstSave.resolve(response({ annotation: savedFrom(bodies[0]), annotations_revision: "labels_1" }));
  await secondStarted.promise;
  assert.equal(ui.api.app.index, 0, "older saved values must not trigger navigation");
  assert.equal(bodies[1].choice, "B");
  assert.equal(bodies[1].reason, "B is clearer and grounded.");
  assert.equal(bodies[1].expected_annotation.pairwise_result.choice, "A");
  secondSave.resolve(response({ annotation: savedFrom(bodies[1]), annotations_revision: "labels_2" }));
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 1);
  assert.equal(bodies.length, 2, "queued obsolete saves must not write the next pair");
  assert.equal(ui.api.app.annotations.pair_0.pairwise_result.choice, "B");
  assert.equal(ui.api.app.annotations.pair_0.pairwise_result.reason, "B is clearer and grounded.");
});

test("the final successful annotation stays visible and reports completion", async () => {
  const ui = harness();
  ui.setFetch(async (_url, options = {}) => options.method === "PUT"
    ? response({ annotation: savedFrom(JSON.parse(options.body)), annotations_revision: "labels_1" })
    : response(state()));
  await ui.api.load();
  ui.api.choose("B");
  await ui.api.app.saveQueue;
  assert.equal(ui.api.app.index, 0);
  assert.equal(ui.element("done-count").textContent, "1");
  assert.equal(ui.element("choose-b").getAttribute("aria-pressed"), "true");
  assert.equal(ui.element("toast").textContent, "All pairs are labeled");
  assert.equal(ui.element("toast").hidden, false);
  assert.equal(ui.api.app.advanceAfterSavePairId, null);
});
