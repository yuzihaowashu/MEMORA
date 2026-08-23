(() => {
  "use strict";

  const data = window.MEMORA_BENCHMARK;
  if (!data) {
    document.getElementById("item-content").innerHTML = '<div class="empty-state"><p>Benchmark data could not be loaded.</p></div>';
    return;
  }

  const modeConfig = {
    eamQa: {
      label: "EAM-QA",
      source: "https://github.com/yuzihaowashu/MEMORA/tree/main/src/memora_bench/eam_qa/questions",
      typeLabels: { SPref: "Preference", SHabit: "Habit", SRoutine: "Routine", ERecall: "Episode recall" },
    },
    planningReplay: {
      label: "Planning Replay",
      source: "https://github.com/yuzihaowashu/MEMORA/tree/main/src/memora_bench/planning/suites/replay",
      typeLabels: {},
    },
    planningGeneralize: {
      label: "Planning Generalize",
      source: "https://github.com/yuzihaowashu/MEMORA/tree/main/src/memora_bench/planning/suites/generalize",
      typeLabels: {},
    },
  };

  const elements = {
    tabs: [...document.querySelectorAll(".protocol-tab")],
    participant: document.getElementById("participant-filter"),
    type: document.getElementById("type-filter"),
    typeLabel: document.getElementById("type-filter-label"),
    typeNote: document.getElementById("type-filter-note"),
    includeControls: document.getElementById("include-controls"),
    controlWrap: document.getElementById("control-toggle-wrap"),
    count: document.getElementById("filtered-count"),
    random: document.getElementById("random-item"),
    source: document.getElementById("source-link"),
    position: document.getElementById("item-position"),
    id: document.getElementById("item-id"),
    content: document.getElementById("item-content"),
    previous: document.getElementById("previous-item"),
    next: document.getElementById("next-item"),
    copy: document.getElementById("copy-item-link"),
    copyBibtex: document.getElementById("copy-bibtex"),
    bibtex: document.getElementById("bibtex-entry"),
  };

  const query = new URLSearchParams(window.location.search);
  const requestedMode = query.get("mode");
  let mode = Object.hasOwn(modeConfig, requestedMode) ? requestedMode : "eamQa";
  let filteredItems = [];
  let itemIndex = 0;
  let requestedId = query.get("id");
  let requestedType = query.get("type");

  const escapeHtml = (value) => String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

  const titleCase = (value) => String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());

  function renderVideoEvidence(label, videos, emptyText = "No supporting recording is associated with this item.", className = "") {
    const uniqueVideos = [...new Set((videos || []).filter(Boolean))];
    const classes = `video-evidence ${className}`.trim();
    if (!uniqueVideos.length) {
      return `<div class="${classes} empty-evidence"><span>${escapeHtml(label)}</span><p>${escapeHtml(emptyText)}</p></div>`;
    }
    const chips = uniqueVideos.map((video) => `<span class="video-chip">${escapeHtml(video)}</span>`).join("");
    if (uniqueVideos.length <= 5) {
      return `<div class="${classes}"><span>${escapeHtml(label)} · ${uniqueVideos.length}</span><div class="video-list">${chips}</div></div>`;
    }
    return `<details class="${classes}"><summary>${escapeHtml(label)} · ${uniqueVideos.length}</summary><div class="video-list">${chips}</div></details>`;
  }

  function setStaticStats() {
    document.querySelectorAll("[data-stat]").forEach((node) => {
      const value = data.stats[node.dataset.stat];
      if (value !== undefined) node.textContent = Number(value).toLocaleString();
    });
    document.querySelectorAll("[data-type-count]").forEach((node) => {
      const value = data.stats.qaTypes[node.dataset.typeCount];
      if (value !== undefined) node.textContent = `${Number(value).toLocaleString()} items`;
    });
  }

  function populateParticipants() {
    data.participants.forEach((participant) => {
      const option = document.createElement("option");
      option.value = participant;
      option.textContent = participant;
      elements.participant.append(option);
    });
    const requestedParticipant = query.get("participant");
    if (requestedParticipant && data.participants.includes(requestedParticipant)) {
      elements.participant.value = requestedParticipant;
    }
  }

  function currentItems() {
    return data[mode];
  }

  function updateTypeOptions() {
    const selected = elements.type.value;
    const participant = elements.participant.value;
    const availableItems = currentItems().filter((item) => participant === "all" || item.participant === participant);
    const typeCounts = availableItems.reduce((counts, item) => {
      counts.set(item.type, (counts.get(item.type) || 0) + 1);
      return counts;
    }, new Map());
    const types = [...typeCounts.keys()].sort();
    elements.type.innerHTML = `<option value="all">All categories (${availableItems.length.toLocaleString()})</option>`;
    types.forEach((type) => {
      const option = document.createElement("option");
      option.value = type;
      const label = modeConfig[mode].typeLabels[type] || titleCase(type);
      option.textContent = `${label} (${typeCounts.get(type).toLocaleString()})`;
      elements.type.append(option);
    });
    if (types.includes(selected)) elements.type.value = selected;
    if (requestedType && types.includes(requestedType)) elements.type.value = requestedType;
    requestedType = null;
    elements.typeLabel.textContent = mode === "eamQa" ? "EAM type" : "Task category";
    elements.typeNote.textContent = "Defined in MEMORA-Bench";
  }

  function updateUrl(item) {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("mode", mode);
      url.searchParams.set("id", item.id);
      if (elements.participant.value === "all") url.searchParams.delete("participant");
      else url.searchParams.set("participant", elements.participant.value);
      if (elements.type.value === "all") url.searchParams.delete("type");
      else url.searchParams.set("type", elements.type.value);
      window.history.replaceState({}, "", url);
    } catch (_error) {
      // Local file previews may not permit history updates.
    }
  }

  function applyFilters({ preserveId = true } = {}) {
    const participant = elements.participant.value;
    const type = elements.type.value;
    const currentId = preserveId && filteredItems[itemIndex] ? filteredItems[itemIndex].id : requestedId;

    filteredItems = currentItems().filter((item) => {
      if (participant !== "all" && item.participant !== participant) return false;
      if (type !== "all" && item.type !== type) return false;
      if (mode === "eamQa" && !elements.includeControls.checked && item.isUnanswerableControl) return false;
      return true;
    });

    const matchingIndex = currentId ? filteredItems.findIndex((item) => item.id === currentId) : -1;
    itemIndex = matchingIndex >= 0 ? matchingIndex : 0;
    requestedId = null;
    elements.count.textContent = filteredItems.length.toLocaleString();
    renderItem();
  }

  function renderQa(item) {
    const typeLabel = modeConfig.eamQa.typeLabels[item.type] || item.type;
    const choices = item.choices.map((choice, index) => {
      const letter = String.fromCharCode(65 + index);
      return `<button class="choice-button" type="button" data-choice="${letter}"><span class="choice-letter">${letter}</span><span class="choice-text">${escapeHtml(choice)}</span></button>`;
    }).join("");

    elements.content.className = "item-content qa-mode";
    elements.content.innerHTML = `
      <div class="item-meta">
        <span class="meta-chip primary">${escapeHtml(typeLabel)}</span>
        <span class="meta-chip">${escapeHtml(item.participant)}</span>
        ${item.isUnanswerableControl ? '<span class="meta-chip control">No-evidence control</span>' : ""}
      </div>
      <p class="item-prompt-label">Question</p>
      <h2 class="item-question">${escapeHtml(item.question)}</h2>
      ${renderVideoEvidence(
        "Supporting recordings",
        item.videoIds,
        item.isUnanswerableControl
          ? "No supporting recording exists for this control; the expected behavior is to abstain."
          : undefined,
      )}
      <div class="choice-stack">${choices}</div>
      <div class="answer-actions">
        <button class="answer-button" type="button" id="check-answer">Check selected answer</button>
        <button class="answer-button secondary" type="button" id="reveal-answer">Reveal ground truth</button>
      </div>
      <div class="answer-panel" id="answer-panel" hidden>
        <span>Ground truth</span>
        <strong>${escapeHtml(item.correctAnswer)}. ${escapeHtml(item.groundTruth)}</strong>
      </div>`;

    let selectedChoice = null;
    const choiceButtons = [...elements.content.querySelectorAll(".choice-button")];
    choiceButtons.forEach((button) => {
      button.addEventListener("click", () => {
        choiceButtons.forEach((candidate) => candidate.classList.remove("selected", "correct", "incorrect"));
        selectedChoice = button.dataset.choice;
        button.classList.add("selected");
      });
    });

    const reveal = () => {
      document.getElementById("answer-panel").hidden = false;
      choiceButtons.forEach((button) => {
        button.classList.remove("selected");
        if (button.dataset.choice === item.correctAnswer) button.classList.add("correct");
        else if (button.dataset.choice === selectedChoice) button.classList.add("incorrect");
      });
    };
    document.getElementById("check-answer").addEventListener("click", reveal);
    document.getElementById("reveal-answer").addEventListener("click", reveal);
  }

  function renderPlanning(item) {
    const isGeneralize = mode === "planningGeneralize";
    const context = [];
    if (item.primaryObjects?.length) context.push(`Objects: ${item.primaryObjects.join(", ")}`);
    if (item.sourceAction) context.push(`Remembered action: ${item.sourceAction}`);
    if (item.sourceObject && item.targetObject) context.push(`Transfer: ${item.sourceObject} to ${item.targetObject}`);
    const steps = item.groundTruthSteps.map((step) => `<li>${escapeHtml(step)}</li>`).join("");
    const videoEvidence = isGeneralize
      ? renderVideoEvidence("Participant experience forming this memory", item.videoIds, undefined, "memory-evidence")
      : `<div class="planning-video-evidence">
          ${renderVideoEvidence("Observed workflow recording", item.video ? [item.video] : [], undefined, "source-evidence")}
          ${renderVideoEvidence("Participant experience forming this memory", item.videoIds, undefined, "memory-evidence")}
        </div>`;

    elements.content.className = `item-content ${isGeneralize ? "generalize-mode" : "replay-mode"}`;
    elements.content.innerHTML = `
      <div class="item-meta">
        <span class="meta-chip primary">${isGeneralize ? "Generalize" : "Replay"}</span>
        <span class="meta-chip">${escapeHtml(titleCase(item.type))}</span>
        <span class="meta-chip">${escapeHtml(item.participant)}</span>
      </div>
      <p class="item-prompt-label">Future goal</p>
      <h2 class="planning-goal">${escapeHtml(item.query)}</h2>
      ${context.length ? `<div class="planning-context">${context.map((value) => `<span class="context-chip">${escapeHtml(value)}</span>`).join("")}</div>` : ""}
      ${videoEvidence}
      <div class="plan-heading"><h3>Reference plan</h3><span>${item.groundTruthSteps.length} steps</span></div>
      <ol class="reference-plan">${steps}</ol>
      ${item.rationale ? `<div class="planning-rationale"><span>Construction rationale</span><p>${escapeHtml(item.rationale)}</p></div>` : ""}`;
  }

  function renderItem() {
    if (!filteredItems.length) {
      elements.position.textContent = "No matching items";
      elements.id.textContent = "Change a filter to continue";
      elements.content.innerHTML = '<div class="empty-state"><p>No benchmark items match the current filters.</p></div>';
      elements.previous.disabled = true;
      elements.next.disabled = true;
      return;
    }

    const item = filteredItems[itemIndex];
    elements.position.textContent = `Item ${itemIndex + 1} of ${filteredItems.length}`;
    elements.id.textContent = item.id;
    elements.previous.disabled = filteredItems.length < 2;
    elements.next.disabled = filteredItems.length < 2;
    if (mode === "eamQa") renderQa(item);
    else renderPlanning(item);
    updateUrl(item);
  }

  function setMode(nextMode) {
    mode = nextMode;
    elements.tabs.forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    elements.controlWrap.hidden = mode !== "eamQa";
    elements.source.href = modeConfig[mode].source;
    updateTypeOptions();
    applyFilters({ preserveId: false });
  }

  function copyCurrentLink() {
    const value = window.location.href;
    const markCopied = () => {
      const original = elements.copy.textContent;
      elements.copy.textContent = "Copied";
      window.setTimeout(() => { elements.copy.textContent = original; }, 1200);
    };
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(value).then(markCopied);
    else {
      const input = document.createElement("textarea");
      input.value = value;
      document.body.append(input);
      input.select();
      document.execCommand("copy");
      input.remove();
      markCopied();
    }
  }

  function copyBibtex() {
    if (!elements.copyBibtex || !elements.bibtex) return;
    const value = elements.bibtex.textContent;
    const markCopied = () => {
      elements.copyBibtex.textContent = "Copied";
      window.setTimeout(() => { elements.copyBibtex.textContent = "Copy BibTeX"; }, 1200);
    };
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(value).then(markCopied);
    else {
      const input = document.createElement("textarea");
      input.value = value;
      document.body.append(input);
      input.select();
      document.execCommand("copy");
      input.remove();
      markCopied();
    }
  }

  elements.tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  elements.participant.addEventListener("change", () => {
    updateTypeOptions();
    applyFilters({ preserveId: false });
  });
  elements.type.addEventListener("change", () => applyFilters({ preserveId: false }));
  elements.includeControls.addEventListener("change", () => applyFilters({ preserveId: false }));
  elements.previous.addEventListener("click", () => {
    if (!filteredItems.length) return;
    itemIndex = (itemIndex - 1 + filteredItems.length) % filteredItems.length;
    renderItem();
  });
  elements.next.addEventListener("click", () => {
    if (!filteredItems.length) return;
    itemIndex = (itemIndex + 1) % filteredItems.length;
    renderItem();
  });
  elements.random.addEventListener("click", () => {
    if (!filteredItems.length) return;
    itemIndex = Math.floor(Math.random() * filteredItems.length);
    renderItem();
  });
  elements.copy.addEventListener("click", copyCurrentLink);
  elements.copyBibtex?.addEventListener("click", copyBibtex);

  setStaticStats();
  populateParticipants();
  setMode(mode);
})();
