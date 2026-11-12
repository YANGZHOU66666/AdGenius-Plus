(() => {
  const $ = (id) => document.getElementById(id);
  const els = {
    sync: $("sync-status"), annotatorForm: $("annotator-form"), annotatorInput: $("annotator-input"),
    done: $("done-count"), total: $("total-count"), progressCopy: $("progress-copy"), progressBar: $("progress-bar"),
    export: $("export-button"), notice: $("notice"), noticeTitle: $("notice-title"), noticeMessage: $("notice-message"),
    loadRemote: $("load-remote"), keepLocal: $("keep-local"), retrySave: $("retry-save"),
    loading: $("loading"), loadingMessage: $("loading-message"), retryLoad: $("retry-load"), panel: $("annotation-panel"),
    number: $("question-number"), entryStatus: $("entry-status"), pageNumber: $("page-number"),
    productDetails: $("product-details"), productTitle: $("product-title"), productInfo: $("product-information"),
    cardA: $("card-a"), cardB: $("card-b"), copyA: $("copy-a"), copyB: $("copy-b"),
    chooseA: $("choose-a"), chooseB: $("choose-b"), clear: $("clear-button"),
    reason: $("reason"), reasonHint: $("reason-hint"), saveCaption: $("save-caption"), savedTime: $("saved-time"),
    previous: $("previous"), next: $("next"), nextUnannotated: $("next-unannotated"),
    grid: $("question-grid"), navigatorCount: $("navigator-count"), filePaths: $("file-paths"), toast: $("toast"),
  };

  const app = {
    annotatorId: new URLSearchParams(window.location.search).get("annotator_id") || "zhou",
    index: 0, state: null, annotations: {}, drafts: {},
    annotationsRevision: null, draftsRevision: null, datasetRevision: null,
    baselineAnnotation: null, baselineDraft: null, leftCandidate: null, rightCandidate: null,
    choice: null, reason: "", dirty: false, writePending: 0, navBusy: false,
    conflict: null, saveError: null, debounceTimer: null, saveQueue: Promise.resolve(),
    pollTimer: null, toastTimer: null, pollPending: false, loadPending: false,
    generation: 0, writeEpoch: 0, advanceAfterSavePairId: null,
  };

  function apiError(status, message) {
    const error = new Error(message || `HTTP ${status}`);
    error.status = status;
    return error;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, { cache: "no-store", ...options,
      headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
    });
    let payload = {};
    try { payload = await response.json(); } catch { /* handled below */ }
    if (!response.ok) throw apiError(response.status, payload.error || payload.message);
    return payload;
  }

  function setSync(state, text) {
    els.sync.dataset.state = state;
    els.sync.textContent = text;
  }

  function setNotice(title = "", message = "", actions = []) {
    els.notice.hidden = !title;
    els.noticeTitle.textContent = title;
    els.noticeMessage.textContent = message;
    els.loadRemote.hidden = !actions.includes("load");
    els.keepLocal.hidden = !actions.includes("keep");
    els.retrySave.hidden = !actions.includes("retry");
  }

  function showToast(text) {
    els.toast.textContent = text;
    els.toast.hidden = false;
    clearTimeout(app.toastTimer);
    app.toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2600);
  }

  function selectedId() {
    if (app.choice === "A") return app.leftCandidate?.candidate_id || null;
    if (app.choice === "B") return app.rightCandidate?.candidate_id || null;
    return null;
  }

  function readRemoteActive() {
    return {
      annotation: app.state?.annotations?.[currentPair()?.pair_id] || null,
      draft: app.state?.drafts?.[currentPair()?.pair_id] || null,
    };
  }

  function currentPair() {
    return app.state?.pairs?.[app.index] || null;
  }

  function randomOrientation(pair) {
    if (!pair?.candidates?.length) {
      app.leftCandidate = null;
      app.rightCandidate = null;
      return;
    }
    const candidates = [...pair.candidates];
    if (window.crypto?.getRandomValues) {
      const bit = new Uint8Array(1);
      window.crypto.getRandomValues(bit);
      if (bit[0] >= 128) candidates.reverse();
    } else if (Math.random() >= 0.5) {
      candidates.reverse();
    }
    [app.leftCandidate, app.rightCandidate] = candidates;
  }

  function initPairView({ shuffle = false } = {}) {
    const pair = currentPair();
    if (!pair) return;
    app.advanceAfterSavePairId = null;
    if (shuffle || !app.leftCandidate ||
        !pair.candidates.some((candidate) => candidate.candidate_id === app.leftCandidate.candidate_id)) {
      randomOrientation(pair);
    }

    app.baselineAnnotation = app.annotations[pair.pair_id] || null;
    app.baselineDraft = app.drafts[pair.pair_id] || null;
    app.conflict = null;
    app.dirty = false;
    const annotation = app.baselineAnnotation;
    if (annotation?.pairwise_result?.status === "success") {
      const winner = annotation.pairwise_result.winner_candidate_id;
      app.choice = winner === null ? null : winner === app.leftCandidate.candidate_id ? "A" : "B";
      app.reason = annotation.pairwise_result.reason || "";
    } else {
      app.choice = null;
      app.reason = app.baselineDraft?.reason || "";
    }
    renderPair();
    renderProgress();
    renderGrid();
    setNotice();
  }

  function renderPair() {
    const pair = currentPair();
    if (!pair) return;
    els.panel.hidden = false;
    els.loading.hidden = true;
    els.number.textContent = `PAIR ${String(app.index + 1).padStart(3, "0")}`;
    els.pageNumber.textContent = `${app.index + 1} / ${app.state.pairs.length}`;
    els.productTitle.textContent = pair.product_title || pair.product_id;
    els.productInfo.textContent = pair.product_information || "No additional product information available.";
    els.copyA.textContent = app.leftCandidate.text;
    els.copyB.textContent = app.rightCandidate.text;
    els.chooseA.setAttribute("aria-pressed", String(app.choice === "A"));
    els.chooseB.setAttribute("aria-pressed", String(app.choice === "B"));
    els.cardA.classList.toggle("selected", app.choice === "A");
    els.cardB.classList.toggle("selected", app.choice === "B");
    els.clear.disabled = !app.baselineAnnotation && !app.baselineDraft && !app.choice && !app.reason;
    const complete = Boolean(app.baselineAnnotation?.pairwise_result?.status === "success");
    els.entryStatus.textContent = complete ? "Labeled" : app.baselineDraft?.reason ? "Draft saved" : "Unlabeled";
    els.entryStatus.classList.toggle("completed", complete);
    // A render after an in-flight save must not reset the textarea's caret or
    // discard characters typed after the request's snapshot was taken.
    if (els.reason.value !== app.reason) els.reason.value = app.reason;
    els.reasonHint.textContent = app.choice ? "Reason is saved with your choice" : "Without a choice, your reason is saved as a draft";
    if (app.writePending) {
      els.saveCaption.textContent = "Syncing to local files…";
    } else if (app.dirty) {
      els.saveCaption.textContent = "Changes waiting to sync";
    } else if (complete) {
      els.saveCaption.textContent = "Choice saved to local file";
    } else if (app.baselineDraft?.reason) {
      els.saveCaption.textContent = "Reason draft saved; this pair is still unlabeled";
    } else {
      els.saveCaption.textContent = "This pair is not labeled yet";
    }
    els.savedTime.textContent = complete ? "Saved to annotation JSONL" : "";
    els.previous.disabled = app.index === 0 || app.navBusy;
    els.next.disabled = app.index === app.state.pairs.length - 1 || app.navBusy;
    els.nextUnannotated.disabled = app.navBusy || !app.state.pairs.some((item) =>
      !app.annotations[item.pair_id] && item.pair_id !== pair.pair_id);
    renderProgress();
    updateNavigationLock();
  }

  function renderProgress() {
    if (!app.state) return;
    const total = app.state.pairs.length;
    const completed = app.state.pairs.filter((pair) => app.annotations[pair.pair_id]?.pairwise_result?.status === "success").length;
    const drafts = app.state.pairs.filter((pair) => app.drafts[pair.pair_id]?.reason && !app.annotations[pair.pair_id]).length;
    els.done.textContent = String(completed);
    els.total.textContent = String(total);
    els.progressCopy.textContent = drafts ? `${drafts} unfinished reason drafts` : `${total - completed} pairs remaining`;
    els.progressBar.style.width = `${total ? (completed / total) * 100 : 0}%`;
    els.navigatorCount.textContent = `${completed} / ${total}`;
    els.export.disabled = completed === 0;
  }

  function renderGrid() {
    if (!app.state) return;
    const fragment = document.createDocumentFragment();
    app.state.pairs.forEach((pair, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "question-cell";
      button.textContent = String(index + 1).padStart(2, "0");
      button.setAttribute("aria-label", `Pair ${index + 1}, ${app.annotations[pair.pair_id] ? "Labeled" : app.drafts[pair.pair_id]?.reason ? "Draft saved, unlabeled" : "Unlabeled"}`);
      button.title = `${pair.product_id} · ${app.annotations[pair.pair_id] ? "Labeled" : app.drafts[pair.pair_id]?.reason ? "Draft" : "Unlabeled"}`;
      button.classList.toggle("completed", app.annotations[pair.pair_id]?.pairwise_result?.status === "success");
      button.classList.toggle("has-draft", Boolean(app.drafts[pair.pair_id]?.reason) && !app.annotations[pair.pair_id]);
      button.classList.toggle("current", index === app.index);
      button.addEventListener("click", () => navigateTo(index));
      fragment.append(button);
    });
    els.grid.replaceChildren(fragment);
  }

  function renderPaths() {
    const paths = app.state?.paths || {};
    els.filePaths.replaceChildren();
    Object.entries(paths).forEach(([name, path]) => {
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = name;
      description.textContent = path;
      els.filePaths.append(term, description);
    });
  }

  function annotationEquivalent(a, b) {
    return JSON.stringify(a || null) === JSON.stringify(b || null);
  }

  function draftEquivalent(a, b) {
    return JSON.stringify(a || null) === JSON.stringify(b || null);
  }

  function captureLocalValues() {
    const pair = currentPair();
    if (!pair) return null;
    return { pair_id: pair.pair_id, choice: app.choice, reason: els.reason.value,
      presentation: { A: app.leftCandidate.candidate_id, B: app.rightCandidate.candidate_id } };
  }

  function setConflict(title, message, remote, operation = app.conflict?.operation || "save") {
    app.conflict = { remote, operation };
    app.saveError = null;
    setSync("error", "Resolve sync conflict");
    const actions = remote?.dataset_revision === app.datasetRevision ? ["load", "keep"] : ["load"];
    setNotice(title, message, actions);
    updateNavigationLock();
  }

  function updateNavigationLock() {
    const blocked = app.navBusy || app.loadPending || app.writePending > 0
      || Boolean(app.conflict) || Boolean(app.saveError);
    els.previous.disabled = blocked || app.index === 0;
    els.next.disabled = blocked || app.index === (app.state?.pairs.length || 0) - 1;
    els.nextUnannotated.disabled = blocked || !app.state?.pairs.some((pair) =>
      !app.annotations[pair.pair_id] && pair.pair_id !== currentPair()?.pair_id);
    els.grid.querySelectorAll("button").forEach((button) => { button.disabled = blocked; });
    els.reason.disabled = app.navBusy || app.loadPending || Boolean(app.conflict);
    els.chooseA.disabled = app.navBusy || app.loadPending || Boolean(app.conflict);
    els.chooseB.disabled = app.navBusy || app.loadPending || Boolean(app.conflict);
    els.annotatorInput.disabled = app.loadPending;
    els.annotatorForm.querySelector("button").disabled = app.loadPending;
  }

  async function fetchState() {
    const query = new URLSearchParams({ annotator_id: app.annotatorId });
    return request(`/api/state?${query}`);
  }

  function applyStatePayload(payload, { initial = false, force = false } = {}) {
    const oldDatasetRevision = app.datasetRevision;
    const oldAnnotationRevision = app.annotationsRevision;
    const oldDraftRevision = app.draftsRevision;
    const pair = currentPair();
    const remoteAnnotation = pair ? payload.annotations?.[pair.pair_id] || null : null;
    const remoteDraft = pair ? payload.drafts?.[pair.pair_id] || null : null;
    const datasetChanged = oldDatasetRevision !== null && payload.dataset_revision !== oldDatasetRevision;
    const recordChanged = pair && (!annotationEquivalent(remoteAnnotation, app.baselineAnnotation)
      || !draftEquivalent(remoteDraft, app.baselineDraft));

    // Once a conflict is visible, polling may refresh its remote snapshot but
    // must not make the local/remote choice on behalf of the annotator.
    if (!initial && !force && app.conflict) {
      app.conflict.remote = payload;
      setConflict("The file version has changed", datasetChanged
        ? "Product or candidate files have changed. Load the updated dataset to continue."
        : "Your changes are still here. Load the file version, or save your changes against the latest version.", payload);
      return;
    }

    if (!initial && !force && (recordChanged || datasetChanged) && app.dirty) {
      if (app.writePending) return;
      app.conflict = { remote: payload };
      setSync("error", "Resolve sync conflict");
      const sameDataset = payload.dataset_revision === app.datasetRevision;
      setNotice("This pair was updated elsewhere", sameDataset
        ? "Your changes are still on this page. Load the file version, or save your changes against the latest version."
        : "Product or candidate files have changed. Your changes are still here; load the updated dataset to continue.",
      sameDataset ? ["load", "keep"] : ["load"]);
      renderProgress();
      renderGrid();
      updateNavigationLock();
      return;
    }

    app.state = payload;
    app.annotations = { ...(payload.annotations || {}) };
    app.drafts = { ...(payload.drafts || {}) };
    app.datasetRevision = payload.dataset_revision;
    app.annotationsRevision = payload.annotations_revision;
    app.draftsRevision = payload.drafts_revision;

    if (initial || datasetChanged || force) {
      app.index = Math.min(app.index, Math.max(payload.pairs.length - 1, 0));
      initPairView({ shuffle: !force || datasetChanged });
      renderPaths();
    } else {
      const active = currentPair();
      if (active && recordChanged && !app.dirty) {
        app.baselineAnnotation = app.annotations[active.pair_id] || null;
        app.baselineDraft = app.drafts[active.pair_id] || null;
        const winner = app.baselineAnnotation?.pairwise_result?.winner_candidate_id;
        app.choice = app.baselineAnnotation
          ? (winner === null ? null : winner === app.leftCandidate.candidate_id ? "A" : "B")
          : null;
        app.reason = app.baselineAnnotation?.pairwise_result?.reason
          || app.baselineDraft?.reason || "";
        renderPair();
      } else {
        renderProgress();
        renderGrid();
      }
      renderPaths();
    }

    if (oldDatasetRevision !== null && datasetChanged) {
      app.conflict = null;
      setNotice("Dataset files updated", "The pair list has been reloaded from disk.", []);
    } else if (oldAnnotationRevision !== payload.annotations_revision
        || oldDraftRevision !== payload.drafts_revision) {
      if (!app.dirty && !app.saveError) setNotice();
    }
    if (app.saveError) setSync("error", "Save failed; retry required");
    else if (!app.conflict) setSync("connected", "Synced to local files");
    updateNavigationLock();
  }

  async function syncState({ force = false } = {}) {
    if (app.pollPending || app.loadPending || app.writePending || app.navBusy) return null;
    const generation = app.generation;
    const writeEpoch = app.writeEpoch;
    const annotatorId = app.annotatorId;
    app.pollPending = true;
    try {
    const payload = await fetchState();
    if (generation !== app.generation || writeEpoch !== app.writeEpoch
        || annotatorId !== app.annotatorId || payload.annotator_id !== annotatorId || app.loadPending
        || app.writePending || app.navBusy) return null;
    applyStatePayload(payload, { force });
    return payload;
    } finally {
      app.pollPending = false;
    }
  }

  function startPolling() {
    clearInterval(app.pollTimer);
    app.pollTimer = setInterval(() => {
      syncState().catch((error) => {
        setSync("error", "Local sync is unavailable");
        if (!app.state) showLoadingError(error);
      });
    }, 1000);
  }

  function showLoadingError(error) {
    els.loading.classList.add("error");
    els.loadingMessage.textContent = `Could not load annotation data: ${error.message}. Check that the local server is running.`;
    els.retryLoad.hidden = false;
  }

  async function load({ shuffle = true } = {}) {
    const generation = ++app.generation;
    const annotatorId = app.annotatorId;
    app.loadPending = true;
    clearInterval(app.pollTimer);
    updateNavigationLock();
    els.panel.hidden = true;
    els.loading.hidden = false;
    els.loading.classList.remove("error");
    els.loadingMessage.textContent = "Loading products and candidate ads…";
    els.retryLoad.hidden = true;
    setSync("saving", "Connecting to local files…");
    try {
      const payload = await fetchState();
      if (generation !== app.generation || annotatorId !== app.annotatorId) return;
      app.index = 0;
      app.state = payload;
      app.annotations = { ...(payload.annotations || {}) };
      app.drafts = { ...(payload.drafts || {}) };
      app.datasetRevision = payload.dataset_revision;
      app.annotationsRevision = payload.annotations_revision;
      app.draftsRevision = payload.drafts_revision;
      app.annotatorId = payload.annotator_id;
      app.saveError = null;
      els.annotatorInput.value = app.annotatorId;
      initPairView({ shuffle });
      renderPaths();
      startPolling();
      setSync("connected", "Synced to local files");
      if (!payload.pairs.length) showLoadingError(new Error("No pairs available for annotation"));
    } catch (error) {
      if (generation !== app.generation) return;
      setSync("error", "Could not connect to local files");
      showLoadingError(error);
    } finally {
      if (generation === app.generation) {
        app.loadPending = false;
        updateNavigationLock();
      }
    }
  }

  function getDraftExpected(pairId) {
    return app.drafts[pairId] || null;
  }

  function setSaving(on) {
    app.writeEpoch += 1;
    app.writePending += on ? 1 : -1;
    app.writePending = Math.max(0, app.writePending);
    if (app.writePending) setSync("saving", "Saving to local files…");
    else if (app.saveError) setSync("error", "Save failed; retry required");
    else if (!app.conflict) setSync("connected", "Synced to local files");
    if (currentPair()) renderPair();
    updateNavigationLock();
  }

  async function refreshAfterConflict(operation = "save") {
    try {
      const payload = await fetchState();
      setConflict("This pair was updated elsewhere", "Your changes are still here. Load the disk version, or save your changes against the latest file version.", payload, operation);
    } catch (error) {
      app.saveError = { operation, message: error.message };
      setNotice("Save conflict", `${error.message}. Your changes are still on this page. Reconnect to resolve this.`, ["retry"]);
      setSync("error", "Save conflict needs attention");
    }
  }

  function saveSnapshot() {
    app.saveQueue = app.saveQueue.then(async () => {
      if (!app.state || app.conflict || app.saveError || !app.dirty) return;
      const snapshot = captureLocalValues();
      if (!snapshot) return;
      const pairId = snapshot.pair_id;
      const needsAnnotation = Boolean(snapshot.choice);
      let advance = false;
      setSaving(true);
      try {
        if (needsAnnotation) {
          const payload = await request("/api/annotation", {
            method: "PUT",
            body: JSON.stringify({
              annotator_id: app.annotatorId,
              pair_id: pairId,
              dataset_revision: app.datasetRevision,
              expected_annotation: app.annotations[pairId] || null,
              presentation: snapshot.presentation,
              choice: snapshot.choice,
              reason: snapshot.reason.trim() || null,
            }),
          });
          app.annotations[pairId] = payload.annotation;
          app.baselineAnnotation = payload.annotation;
          app.annotationsRevision = payload.annotations_revision;
          if (payload.drafts_revision !== undefined) app.draftsRevision = payload.drafts_revision;
          if (app.drafts[pairId]) delete app.drafts[pairId];
          app.baselineDraft = null;
        } else if (snapshot.reason.trim()) {
          const payload = await request("/api/draft", {
            method: "PUT",
            body: JSON.stringify({ annotator_id: app.annotatorId, pair_id: pairId,
              dataset_revision: app.datasetRevision,
              expected_draft: getDraftExpected(pairId), reason: snapshot.reason }),
          });
          if (payload.draft) app.drafts[pairId] = payload.draft;
          else delete app.drafts[pairId];
          app.baselineDraft = payload.draft || null;
          app.draftsRevision = payload.drafts_revision;
        } else if (app.drafts[pairId]) {
          const payload = await request("/api/draft", {
            method: "PUT",
            body: JSON.stringify({ annotator_id: app.annotatorId, pair_id: pairId,
              dataset_revision: app.datasetRevision,
              expected_draft: getDraftExpected(pairId), reason: "" }),
          });
          delete app.drafts[pairId];
          app.baselineDraft = null;
          app.draftsRevision = payload.drafts_revision;
        }
        const current = currentPair();
        const sameLocal = current?.pair_id === pairId && app.choice === snapshot.choice
          && els.reason.value === snapshot.reason;
        if (sameLocal) app.dirty = false;
        app.saveError = null;
        setNotice();
        renderProgress();
        renderGrid();
        renderPair();
        advance = needsAnnotation && sameLocal && app.advanceAfterSavePairId === pairId;
        if (!sameLocal) queueSave(0);
      } catch (error) {
        if (error.status === 409) {
          await refreshAfterConflict();
        } else {
          app.saveError = { operation: "save", message: error.message };
          setSync("error", "Save failed");
          setNotice("Could not save to local file", `${error.message}. Your changes are still on this page.`, ["retry"]);
        }
      } finally {
        setSaving(false);
      }
      if (advance && !app.navBusy && !app.dirty && !app.saveError && !app.conflict
          && currentPair()?.pair_id === pairId) {
        app.advanceAfterSavePairId = null;
        const nextIndex = nextUnannotatedIndex();
        // This save already flushed the current values. Calling navigateTo here
        // would wait on this same save queue, so switch the view directly.
        if (nextIndex >= 0) showPair(nextIndex);
        else showToast("All pairs are labeled");
      }
    }).catch((error) => {
      app.saveError = { operation: "save", message: error.message };
      setSync("error", "Save queue error");
      setNotice("Save failed", `${error.message}. Your changes are still on this page.`, ["retry"]);
    });
    return app.saveQueue;
  }

  function queueSave(delay = 320) {
    clearTimeout(app.debounceTimer);
    if (!app.dirty) return;
    if (delay === 0) {
      saveSnapshot();
      return;
    }
    app.debounceTimer = setTimeout(() => saveSnapshot(), delay);
  }

  async function flushCurrent() {
    clearTimeout(app.debounceTimer);
    if (app.conflict) return false;
    if (app.dirty) await saveSnapshot();
    await app.saveQueue;
    if (app.writePending || app.conflict || app.saveError || app.dirty) return false;
    return true;
  }

  function choose(choice) {
    if (!["A", "B"].includes(choice) || app.navBusy || app.conflict) return;
    app.choice = choice;
    app.advanceAfterSavePairId = currentPair()?.pair_id || null;
    app.saveError = null;
    app.dirty = true;
    renderPair();
    queueSave(0);
  }

  function editReason() {
    app.reason = els.reason.value;
    app.dirty = true;
    if (app.navBusy || app.conflict) return;
    app.saveError = null;
    els.saveCaption.textContent = "Reason updated; syncing…";
    queueSave(320);
  }

  async function navigateTo(nextIndex, { wrap = false } = {}) {
    if (!app.state || app.navBusy || app.conflict) return;
    let target = nextIndex;
    if (wrap) target = (target + app.state.pairs.length) % app.state.pairs.length;
    if (target < 0 || target >= app.state.pairs.length || target === app.index) return;
    app.navBusy = true;
    renderPair();
    const saved = await flushCurrent();
    if (!saved) { app.navBusy = false; renderPair(); return; }
    app.navBusy = false;
    showPair(target);
  }

  function showPair(index) {
    app.index = index;
    app.leftCandidate = null;
    initPairView({ shuffle: true });
    renderPair();
    renderGrid();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function nextUnannotatedIndex() {
    if (!app.state) return -1;
    for (let offset = 1; offset < app.state.pairs.length; offset += 1) {
      const index = (app.index + offset) % app.state.pairs.length;
      const pair = app.state.pairs[index];
      if (!app.annotations[pair.pair_id]) return index;
    }
    return -1;
  }

  async function navigateNextUnannotated() {
    if (!app.state) return;
    const index = nextUnannotatedIndex();
    if (index >= 0) return navigateTo(index);
    showToast("All pairs are labeled");
  }

  async function clearAnnotation() {
    if (!app.state || app.navBusy || app.conflict) return;
    const pair = currentPair();
    if (!pair) return;
    app.advanceAfterSavePairId = null;
    clearTimeout(app.debounceTimer);
    app.saveError = null;
    app.navBusy = true;
    renderPair();
    await app.saveQueue;
    setSaving(true);
    try {
      if (app.annotations[pair.pair_id]) {
        const payload = await request("/api/annotation", { method: "DELETE", body: JSON.stringify({
          annotator_id: app.annotatorId, pair_id: pair.pair_id,
          dataset_revision: app.datasetRevision,
          expected_annotation: app.annotations[pair.pair_id],
        }) });
        delete app.annotations[pair.pair_id];
        app.annotationsRevision = payload.annotations_revision;
        // Deleting an annotation also clears its disk draft atomically.
        delete app.drafts[pair.pair_id];
        app.baselineDraft = null;
      }
      if (!app.annotations[pair.pair_id] && app.drafts[pair.pair_id]) {
        const payload = await request("/api/draft", { method: "PUT", body: JSON.stringify({
          annotator_id: app.annotatorId, pair_id: pair.pair_id,
          dataset_revision: app.datasetRevision,
          expected_draft: app.drafts[pair.pair_id], reason: "",
        }) });
        delete app.drafts[pair.pair_id];
        app.draftsRevision = payload.drafts_revision;
      }
      app.baselineAnnotation = null;
      app.baselineDraft = null;
      app.choice = null;
      app.reason = "";
      app.dirty = false;
      els.reason.value = "";
      setNotice();
      renderPair();
      renderGrid();
      showToast("Choice and draft cleared");
    } catch (error) {
      if (error.status === 409) await refreshAfterConflict("clear");
      else {
        app.saveError = { operation: "clear", message: error.message };
        setSync("error", "Clear failed");
        setNotice("Could not clear annotation", `${error.message}. The local file update could not be confirmed.`, ["retry"]);
      }
    } finally {
      setSaving(false);
      app.navBusy = false;
      renderPair();
    }
  }

  function loadRemoteVersion() {
    if (!app.conflict) return;
    const payload = app.conflict.remote;
    const oldPairId = currentPair()?.pair_id;
    const localOrientation = { left: app.leftCandidate, right: app.rightCandidate };
    const datasetChanged = payload.dataset_revision !== app.datasetRevision;
    app.state = { ...app.state, ...payload };
    app.annotations = { ...(payload.annotations || {}) };
    app.drafts = { ...(payload.drafts || {}) };
    app.datasetRevision = payload.dataset_revision;
    app.annotationsRevision = payload.annotations_revision;
    app.draftsRevision = payload.drafts_revision;
    const samePairIndex = app.state.pairs.findIndex((item) => item.pair_id === oldPairId);
    if (datasetChanged) app.index = samePairIndex >= 0 ? samePairIndex : 0;
    const pair = currentPair();
    if (pair && !datasetChanged && localOrientation.left
        && pair.candidates.some((candidate) => candidate.candidate_id === localOrientation.left.candidate_id)) {
      app.leftCandidate = localOrientation.left;
      app.rightCandidate = localOrientation.right;
    } else {
      app.leftCandidate = null;
      randomOrientation(pair);
    }
    if (pair) {
      app.baselineAnnotation = app.annotations[pair.pair_id] || null;
      app.baselineDraft = app.drafts[pair.pair_id] || null;
      const winner = app.baselineAnnotation?.pairwise_result?.winner_candidate_id;
      app.choice = app.baselineAnnotation
        ? (winner === null ? null : winner === app.leftCandidate.candidate_id ? "A" : "B") : null;
      app.reason = app.baselineAnnotation?.pairwise_result?.reason || app.baselineDraft?.reason || "";
      app.dirty = false;
    }
    app.conflict = null;
    setNotice();
    setSync("connected", "Synced to local files");
    renderPair(); renderGrid(); renderProgress(); updateNavigationLock();
  }

  async function keepLocalAndResave() {
    if (!app.conflict) return;
    const payload = app.conflict.remote;
    const operation = app.conflict.operation;
    if (payload.dataset_revision !== app.datasetRevision) return;
    app.state = { ...app.state, ...payload };
    app.annotations = { ...(payload.annotations || {}) };
    app.drafts = { ...(payload.drafts || {}) };
    app.datasetRevision = payload.dataset_revision;
    app.annotationsRevision = payload.annotations_revision;
    app.draftsRevision = payload.drafts_revision;
    const pair = currentPair();
    app.baselineAnnotation = app.annotations[pair.pair_id] || null;
    app.baselineDraft = app.drafts[pair.pair_id] || null;
    if (!app.choice && app.baselineAnnotation?.pairwise_result?.status === "success") {
      const winner = app.baselineAnnotation.pairwise_result.winner_candidate_id;
      app.choice = winner === null ? null
        : winner === app.leftCandidate.candidate_id ? "A" : "B";
    }
    app.conflict = null;
    app.saveError = null;
    if (operation === "clear") {
      setNotice();
      await clearAnnotation();
      return;
    }
    app.dirty = true;
    setNotice();
    setSync("saving", "Saving against the latest file version…");
    queueSave(0);
  }

  async function exportAnnotations() {
    if (!app.state) return;
    const records = app.state.pairs.map((pair) => app.annotations[pair.pair_id]).filter(Boolean);
    if (!records.length) return;
    const blob = new Blob([records.map((record) => JSON.stringify(record)).join("\n") + "\n"],
      { type: "application/x-ndjson;charset=utf-8" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `pairwise_annotations_${app.annotatorId}.jsonl`;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
    showToast(`Exported ${records.length} annotations`);
  }

  async function switchAnnotator(event) {
    event.preventDefault();
    const requested = els.annotatorInput.value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(requested)) {
      showToast("Annotator ID must start with a letter or digit and contain only letters, digits, _ or -");
      els.annotatorInput.value = app.annotatorId;
      return;
    }
    if (requested === app.annotatorId) return;
    if (!(await flushCurrent())) {
      els.annotatorInput.value = app.annotatorId;
      showToast("Resolve the current save status before switching annotators");
      return;
    }
    app.annotatorId = requested;
    app.index = 0;
    app.leftCandidate = null;
    await load({ shuffle: true });
  }

  els.chooseA.addEventListener("click", () => choose("A"));
  els.chooseB.addEventListener("click", () => choose("B"));
  els.reason.addEventListener("input", editReason);
  els.previous.addEventListener("click", () => navigateTo(app.index - 1));
  els.next.addEventListener("click", () => navigateTo(app.index + 1));
  els.nextUnannotated.addEventListener("click", navigateNextUnannotated);
  els.clear.addEventListener("click", clearAnnotation);
  els.export.addEventListener("click", exportAnnotations);
  els.annotatorForm.addEventListener("submit", switchAnnotator);
  els.retryLoad.addEventListener("click", () => load({ shuffle: true }));
  els.loadRemote.addEventListener("click", loadRemoteVersion);
  els.keepLocal.addEventListener("click", keepLocalAndResave);
  els.retrySave.addEventListener("click", () => {
    const operation = app.saveError?.operation;
    app.saveError = null;
    if (operation === "clear") { clearAnnotation(); return; }
    if (!app.dirty) app.dirty = true;
    setNotice();
    queueSave(0);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
    if (event.key === "ArrowLeft" && app.index > 0) navigateTo(app.index - 1);
    if (event.key === "ArrowRight" && app.index < (app.state?.pairs.length || 0) - 1) navigateTo(app.index + 1);
    if (event.key.toLowerCase() === "a") choose("A");
    if (event.key.toLowerCase() === "b") choose("B");
  });

  window.addEventListener("beforeunload", (event) => {
    if (app.dirty || app.writePending) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  load({ shuffle: true });
})();
