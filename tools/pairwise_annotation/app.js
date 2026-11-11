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
    chooseA: $("choose-a"), chooseB: $("choose-b"), chooseTie: $("choose-tie"), clear: $("clear-button"),
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
    generation: 0, writeEpoch: 0,
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
    if (app.choice === "tie") return null;
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
      app.choice = winner === null ? "tie" : winner === app.leftCandidate.candidate_id ? "A" : "B";
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
    els.chooseTie.setAttribute("aria-pressed", String(app.choice === "tie"));
    els.cardA.classList.toggle("selected", app.choice === "A");
    els.cardB.classList.toggle("selected", app.choice === "B");
    els.chooseTie.classList.toggle("selected", app.choice === "tie");
    els.clear.disabled = !app.baselineAnnotation && !app.baselineDraft && !app.choice && !app.reason;
    const complete = Boolean(app.baselineAnnotation?.pairwise_result?.status === "success");
    els.entryStatus.textContent = complete ? "已标注" : app.baselineDraft?.reason ? "草稿已同步" : "未标注";
    els.entryStatus.classList.toggle("completed", complete);
    // A render after an in-flight save must not reset the textarea's caret or
    // discard characters typed after the request's snapshot was taken.
    if (els.reason.value !== app.reason) els.reason.value = app.reason;
    els.reasonHint.textContent = app.choice ? "理由会和结论一起保存" : "未选择结论时，理由保存为草稿";
    if (app.writePending) {
      els.saveCaption.textContent = "正在同步到本地文件…";
    } else if (app.dirty) {
      els.saveCaption.textContent = "有更改等待同步";
    } else if (complete) {
      els.saveCaption.textContent = "结论已保存到本地文件";
    } else if (app.baselineDraft?.reason) {
      els.saveCaption.textContent = "理由草稿已保存；本条仍未标注";
    } else {
      els.saveCaption.textContent = "本条尚未标注";
    }
    els.savedTime.textContent = complete ? "已写入标注 JSONL" : "";
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
    els.progressCopy.textContent = drafts ? `${drafts} 条有未完成的理由草稿` : `${total - completed} 对待标注`;
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
      button.setAttribute("aria-label", `第 ${index + 1} 题，${app.annotations[pair.pair_id] ? "已标注" : app.drafts[pair.pair_id]?.reason ? "有草稿，未标注" : "未标注"}`);
      button.title = `${pair.product_id} · ${app.annotations[pair.pair_id] ? "已标注" : app.drafts[pair.pair_id]?.reason ? "草稿" : "未标注"}`;
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
    setSync("error", "需要处理同步冲突");
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
    els.chooseTie.disabled = app.navBusy || app.loadPending || Boolean(app.conflict);
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
      setConflict("本地文件中的版本已变化", datasetChanged
        ? "商品或候选文件已变化。请加载新数据集版本后继续。"
        : "你的当前修改仍保留。请选择加载文件版本，或基于最新文件版本重新保存。", payload);
      return;
    }

    if (!initial && !force && (recordChanged || datasetChanged) && app.dirty) {
      if (app.writePending) return;
      app.conflict = { remote: payload };
      setSync("error", "需要处理同步冲突");
      const sameDataset = payload.dataset_revision === app.datasetRevision;
      setNotice("此条在其他位置已更新", sameDataset
        ? "你的修改仍保留在页面中。请选择加载文件版本，或将当前修改基于最新版本重新保存。"
        : "商品或候选文件已变化。你的当前修改仍保留；请加载新数据集版本后继续。",
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
          ? (winner === null ? "tie" : winner === app.leftCandidate.candidate_id ? "A" : "B")
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
      setNotice("数据集文件已更新", "条目列表已从磁盘重新载入。", []);
    } else if (oldAnnotationRevision !== payload.annotations_revision
        || oldDraftRevision !== payload.drafts_revision) {
      if (!app.dirty && !app.saveError) setNotice();
    }
    if (app.saveError) setSync("error", "保存失败，等待重试");
    else if (!app.conflict) setSync("connected", "已同步到本地文件");
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
        setSync("error", "本地同步暂不可用");
        if (!app.state) showLoadingError(error);
      });
    }, 1000);
  }

  function showLoadingError(error) {
    els.loading.classList.add("error");
    els.loadingMessage.textContent = `无法读取标注数据：${error.message}。请确认本地服务已启动。`;
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
    els.loadingMessage.textContent = "正在读取本地商品与候选文案…";
    els.retryLoad.hidden = true;
    setSync("saving", "连接本地文件…");
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
      setSync("connected", "已同步到本地文件");
      if (!payload.pairs.length) showLoadingError(new Error("没有可标注的 pair"));
    } catch (error) {
      if (generation !== app.generation) return;
      setSync("error", "无法连接本地文件");
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
    if (app.writePending) setSync("saving", "正在写入本地文件…");
    else if (app.saveError) setSync("error", "保存失败，等待重试");
    else if (!app.conflict) setSync("connected", "已同步到本地文件");
    if (currentPair()) renderPair();
    updateNavigationLock();
  }

  async function refreshAfterConflict(operation = "save") {
    try {
      const payload = await fetchState();
      setConflict("此条在其他位置已更新", "你的当前修改仍保留。选择加载磁盘版本，或基于最新文件版本重新保存。", payload, operation);
    } catch (error) {
      app.saveError = { operation, message: error.message };
      setNotice("保存时发生冲突", `${error.message}。当前页面修改仍保留，请重新连接后处理。`, ["retry"]);
      setSync("error", "保存冲突待处理");
    }
  }

  function saveSnapshot() {
    app.saveQueue = app.saveQueue.then(async () => {
      if (!app.state || app.conflict || app.saveError || !app.dirty) return;
      const snapshot = captureLocalValues();
      if (!snapshot) return;
      const pairId = snapshot.pair_id;
      const needsAnnotation = Boolean(snapshot.choice);
      setSaving(true);
      try {
        if (needsAnnotation) {
          const winnerId = snapshot.choice === "tie" ? null
            : snapshot.choice === "A" ? snapshot.presentation.A : snapshot.presentation.B;
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
        if (!sameLocal) queueSave(0);
      } catch (error) {
        if (error.status === 409) {
          await refreshAfterConflict();
        } else {
          app.saveError = { operation: "save", message: error.message };
          setSync("error", "保存失败");
          setNotice("未能保存到本地文件", `${error.message}。你的修改仍保留在当前页面。`, ["retry"]);
        }
      } finally {
        setSaving(false);
      }
    }).catch((error) => {
      app.saveError = { operation: "save", message: error.message };
      setSync("error", "保存队列发生错误");
      setNotice("保存失败", `${error.message}。你的修改仍保留在当前页面。`, ["retry"]);
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
    if (app.navBusy || app.conflict) return;
    app.choice = choice;
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
    els.saveCaption.textContent = "理由已更改，正在同步…";
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
    app.index = target;
    app.leftCandidate = null;
    initPairView({ shuffle: true });
    app.navBusy = false;
    renderPair();
    renderGrid();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function navigateNextUnannotated() {
    if (!app.state) return;
    for (let offset = 1; offset < app.state.pairs.length; offset += 1) {
      const index = (app.index + offset) % app.state.pairs.length;
      const pair = app.state.pairs[index];
      if (!app.annotations[pair.pair_id]) return navigateTo(index);
    }
    showToast("所有 pair 都已标注");
  }

  async function clearAnnotation() {
    if (!app.state || app.navBusy || app.conflict) return;
    const pair = currentPair();
    if (!pair) return;
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
      showToast("本条结论和草稿已清除");
    } catch (error) {
      if (error.status === 409) await refreshAfterConflict("clear");
      else {
        app.saveError = { operation: "clear", message: error.message };
        setSync("error", "清除失败");
        setNotice("未能清除", `${error.message}。本地文件未确认更新。`, ["retry"]);
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
        ? (winner === null ? "tie" : winner === app.leftCandidate.candidate_id ? "A" : "B") : null;
      app.reason = app.baselineAnnotation?.pairwise_result?.reason || app.baselineDraft?.reason || "";
      app.dirty = false;
    }
    app.conflict = null;
    setNotice();
    setSync("connected", "已同步到本地文件");
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
      app.choice = winner === null ? "tie"
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
    setSync("saving", "基于文件新版本保存…");
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
    showToast(`已导出 ${records.length} 条已标注结果`);
  }

  async function switchAnnotator(event) {
    event.preventDefault();
    const requested = els.annotatorInput.value.trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(requested)) {
      showToast("标注者 ID 需以字母或数字开头，后续可使用字母、数字、_ 或 -");
      els.annotatorInput.value = app.annotatorId;
      return;
    }
    if (requested === app.annotatorId) return;
    if (!(await flushCurrent())) {
      els.annotatorInput.value = app.annotatorId;
      showToast("请先解决当前保存状态，再切换标注者");
      return;
    }
    app.annotatorId = requested;
    app.index = 0;
    app.leftCandidate = null;
    await load({ shuffle: true });
  }

  els.chooseA.addEventListener("click", () => choose("A"));
  els.chooseB.addEventListener("click", () => choose("B"));
  els.chooseTie.addEventListener("click", () => choose("tie"));
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
    if (event.key.toLowerCase() === "t") choose("tie");
  });

  window.addEventListener("beforeunload", (event) => {
    if (app.dirty || app.writePending) {
      event.preventDefault();
      event.returnValue = "";
    }
  });

  load({ shuffle: true });
})();
