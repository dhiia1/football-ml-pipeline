/* ==========================================================================
   Pitchcast — frontend logic (vanilla JS, no build step)
   ==========================================================================

   ┌────────────────────────────────────────────────────────────────────────┐
   │  CONFIGURATION — these are the only two values you normally change.    │
   └────────────────────────────────────────────────────────────────────────┘

   API_BASE  Base URL of the FastAPI service, no trailing slash.
             · local dev            "http://localhost:8001"
             · docker-compose       the browser talks to the *published host
                                    port*, not the compose service name, so
                                    "http://localhost:8001" still applies.
                                    Service names like "http://api:8001" only
                                    resolve inside the compose network.
             · nginx reverse proxy  "/api"  — best option for compose: add a
                                    `location /api/ { proxy_pass http://api:8001/; }`
                                    block to nginx.conf and this becomes
                                    same-origin, so no CORS headers needed.
             Anything cross-origin requires CORSMiddleware on the FastAPI side.

   LOGO_DIR  Folder holding "<slugified-team-name>.png" (e.g. fc-barcelona.png,
             real-madrid-cf.png). Relative by default so the app works from any
             path; use "/assets/logos" if it is always served from the root.
             Missing files are expected — they fall back to an initials badge.
   ========================================================================== */

const API_BASE = "https://football-ml-pipeline-api.onrender.com";
//const API_BASE = "http://localhost:8000";
const LOGO_DIR = "assets/logos";
// Explicit filename overrides — "footylogos" filenames don't match the
// slugify() pattern (they use casual names, not official ones), so we
// map each exact API team name straight to its real downloaded file
// instead of trying to guess/rename to fit.
const LOGO_FILENAME_OVERRIDES = {
  "Athletic Club": "athletic-club-bilbao-logo-footylogos.svg",
  "Club Atlético de Madrid": "atletico-madrid-logo-footylogos.svg",
  "RC Celta de Vigo": "celta-vigo-logo-footylogos.svg",
  "Deportivo Alavés": "deportivo-alaves-logo-footylogos.svg",
  "RC Deportivo La Coruña": "deportivo-la-coruna-logo-footylogos.svg",
  "Elche CF": "elche-cf-logo-footylogos.svg",
  "FC Barcelona": "fc-barcelona-logo-footylogos.svg",
  "Getafe CF": "getafe-cf-logo-footylogos.svg",
  "Levante UD": "levante-ud-logo-footylogos.svg",
  "Málaga CF": "malaga-cf-logo-footylogos.svg",
  "CA Osasuna": "osasuna-logo-footylogos.svg",
  "Real Racing Club de Santander": "racing-santander-logo-footylogos.svg",
  "Rayo Vallecano de Madrid": "rayo-vallecano-logo-footylogos.svg",
  "RCD Espanyol de Barcelona": "rcd-espanyol-barcelona-logo-footylogos.svg",
  "Real Betis Balompié": "real-betis-balompie-logo-footylogos.svg",
  "Real Madrid CF": "real-madrid-logo-footylogos.svg",
  "Real Sociedad de Fútbol": "real-sociedad-logo-footylogos.svg",
  "Sevilla FC": "sevilla-fc-logo-footylogos.svg",
  "Valencia CF": "valencia-cf-logo-footylogos.svg",
  "Villarreal CF": "villarreal-cf-logo-footylogos.svg",
};

/* Range offered in the matchday picker. */
const MATCHDAY_MIN = 1;
const MATCHDAY_MAX = 38;


/* ── tiny DOM helper ─────────────────────────────────────────────────────
   h("div", { class: "x", text: "hi" }, child, child) → HTMLElement
   Team names always go in via `text`, never innerHTML.                     */
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "svg") node.innerHTML = value;           // trusted, local only
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

const $ = (selector, root = document) => root.querySelector(selector);

function fill(container, ...nodes) {
  container.replaceChildren(...nodes.flat().filter(Boolean));
}


/* ── API layer ───────────────────────────────────────────────────────── */

class ApiError extends Error {
  constructor(message, { status = 0, detail = null } = {}) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function apiGet(path) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  } catch {
    // DNS failure, connection refused, CORS rejection — all land here.
    throw new ApiError("offline", { status: 0 });
  }

  if (!response.ok) {
    let detail = null;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch { /* body was not JSON — nothing to add */ }
    throw new ApiError(`HTTP ${response.status}`, { status: response.status, detail });
  }

  return response.json();
}

const api = {
  teams: () => apiGet("/teams"),
  round: (matchday) => apiGet(`/round/${encodeURIComponent(matchday)}`),
  game: (home, away) =>
    apiGet(`/game?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`),
  nextFor: (team) => apiGet(`/team/${encodeURIComponent(team)}/next`),
};


/* ── formatting helpers ──────────────────────────────────────────────── */

/** "Atlético de Madrid" → "atletico-de-madrid" (matches the PNG filenames) */
function slugify(name) {
  return String(name)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const SKIP_WORDS = new Set(["de", "del", "la", "el", "los", "of", "the", "and", "y"]);

/** "FC Barcelona" → "FCB", "CA Osasuna" → "CAO", "Athletic Club" → "AC" */
function initials(name) {
  const words = String(name)
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z\s]/g, " ")
    .split(/\s+/)
    .filter((word) => word && !SKIP_WORDS.has(word.toLowerCase()));

  if (!words.length) return "?";

  const letters = words
    .map((word) => (word.length <= 3 && word === word.toUpperCase() ? word : word[0]))
    .join("")
    .toUpperCase();

  return letters.slice(0, 3) || "?";
}

/* Hues chosen to sit comfortably beside the pitch green and the data colours. */
const BADGE_HUES = [8, 32, 96, 152, 172, 205, 264, 336];

function badgeHue(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return BADGE_HUES[hash % BADGE_HUES.length];
}

/** Local-timezone kick-off, e.g. { day: "Sat 10 Oct", time: "18:30" } */
function formatKickoff(iso) {
  if (!iso) return null;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return null;
  return {
    day: new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" }).format(date),
    time: new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(date),
  };
}

const OUTCOMES = {
  "Home Win": { key: "home", label: "Home win", glyph: "1" },
  "Draw":     { key: "draw", label: "Draw",     glyph: "X" },
  "Away Win": { key: "away", label: "Away win", glyph: "2" },
};

function outcomeOf(prediction) {
  return OUTCOMES[prediction] ?? { key: "draw", label: String(prediction ?? "Unknown"), glyph: "?" };
}

/**
 * Whole percentages that always add up to 100 (largest-remainder method), so
 * the labels never read "8 + 4 + 89". Order: away, draw, home.
 */
function toPercents(probabilities = {}) {
  const raw = ["Away Win", "Draw", "Home Win"].map((key) => {
    const value = Number(probabilities?.[key]);
    return Number.isFinite(value) && value > 0 ? value : 0;
  });

  const total = raw.reduce((sum, value) => sum + value, 0);
  if (!total) return { away: 0, draw: 0, home: 0 };

  const scaled = raw.map((value) => (value / total) * 100);
  const result = scaled.map(Math.floor);
  const missing = 100 - result.reduce((sum, value) => sum + value, 0);

  scaled
    .map((value, index) => ({ index, fraction: value - Math.floor(value) }))
    .sort((a, b) => b.fraction - a.fraction)
    .slice(0, Math.max(0, missing))
    .forEach(({ index }) => { result[index] += 1; });

  return { away: result[0], draw: result[1], home: result[2] };
}


/* ── pieces of the match card ────────────────────────────────────────── */

function initialsBadge(teamName) {
  const hue = badgeHue(teamName);
  return h("span", {
    class: "crest-badge",
    "aria-hidden": "true",
    text: initials(teamName),
    style: `background:hsl(${hue} 58% 93%);border-color:hsl(${hue} 46% 78%);color:hsl(${hue} 58% 31%)`,
  });
}

/**
 * Club crest. If the PNG is missing the element swaps itself for an initials
 * badge — a deliberate piece of the design, not a broken-image state.
 */
function crest(teamName) {
  const img = h("img", {
    class: "crest",
    src: `${LOGO_DIR}/${LOGO_FILENAME_OVERRIDES[teamName] || `${slugify(teamName)}.svg`}`,
    alt: "",
    decoding: "async",
  });
  img.addEventListener("error", () => img.replaceWith(initialsBadge(teamName)), { once: true });
  return img;
}

function verdictBand(outcome, percents, { showPercent = false } = {}) {
  const confidence = percents[outcome.key];
  return h("div", { class: `verdict verdict-${outcome.key}` },
    h("span", { class: "verdict-label", text: outcome.label }),
    showPercent && Number.isFinite(confidence)
      ? h("span", { class: "verdict-pct", text: `${confidence}%` })
      : null,
    h("span", { class: "verdict-glyph", "aria-hidden": "true", text: outcome.glyph }),
  );
}

/** Thin stacked bar (away · draw · home) plus the numbers underneath. */
function probabilityBar(percents) {
  const parts = [
    { key: "away", name: "Away win", value: percents.away },
    { key: "draw", name: "Draw",     value: percents.draw },
    { key: "home", name: "Home win", value: percents.home },
  ];

  const bar = h("div", { class: "bar", role: "img",
    "aria-label": parts.map((p) => `${p.name} ${p.value}%`).join(", ") },
    parts.map((part) =>
      h("span", {
        class: `bar-seg seg-${part.key}`,
        // a 1% segment would otherwise be sub-pixel; a 0% one stays invisible
        style: `width:${part.value}%${part.value > 0 ? ";min-width:3px" : ""}`,
        title: `${part.name} ${part.value}%`,
      }),
    ),
  );

  const legend = h("div", { class: "legend", "aria-hidden": "true" },
    parts.map((part) =>
      h("div", { class: "legend-item" },
        h("span", { class: "legend-pct" },
          h("span", { class: "legend-dot", style: `background:var(--${part.key})` }),
          h("span", { text: `${part.value}%` }),
        ),
        h("span", { class: "legend-name", text: part.name.replace(" win", "") }),
      ),
    ),
  );

  return [bar, legend];
}

/**
 * The match card, reused everywhere.
 * variant: "compact" (round grid) | "featured" (head to head, next match)
 */
function matchCard(match, { variant = "compact" } = {}) {
  const outcome = outcomeOf(match.prediction);
  const percents = toPercents(match.probabilities);
  const kickoff = formatKickoff(match.date);
  const featured = variant === "featured";

  const head = h("div", { class: "card-head" },
    kickoff
      ? h("span", { class: "kickoff", text: `${kickoff.day}, ${kickoff.time}` })
      : h("span", { class: "kickoff kickoff-none", text: "Hypothetical matchup" }),
    Number.isFinite(Number(match.matchday))
      ? h("span", { class: "md-box" },
          h("span", { text: "Matchday" }),
          h("b", { text: String(match.matchday) }),
        )
      : null,
  );

  const teams = featured
    ? h("div", { class: "matchup" },
        h("div", { class: `matchup-side${outcome.key === "home" ? " is-favoured" : ""}` },
          crest(match.home_team),
          h("span", { class: "matchup-name", text: match.home_team }),
          h("span", { class: "matchup-role", text: "Home" }),
        ),
        h("span", { class: "matchup-vs", text: "versus" }),
        h("div", { class: `matchup-side${outcome.key === "away" ? " is-favoured" : ""}` },
          crest(match.away_team),
          h("span", { class: "matchup-name", text: match.away_team }),
          h("span", { class: "matchup-role", text: "Away" }),
        ),
      )
    : h("div", { class: "teams" },
        h("div", { class: `team${outcome.key === "home" ? " is-favoured" : ""}` },
          crest(match.home_team),
          h("span", { class: "team-name", text: match.home_team }),
          h("span", { class: "team-side", text: "Home" }),
        ),
        h("div", { class: `team${outcome.key === "away" ? " is-favoured" : ""}` },
          crest(match.away_team),
          h("span", { class: "team-name", text: match.away_team }),
          h("span", { class: "team-side", text: "Away" }),
        ),
      );

  return h("article", { class: `card${featured ? " card-featured" : ""}` },
    head,
    teams,
    verdictBand(outcome, percents, { showPercent: featured }),
    probabilityBar(percents),
  );
}


/* ── loading / empty / error states ──────────────────────────────────── */

function skeleton({ featured = false } = {}) {
  return h("div", { class: `sk${featured ? " sk-featured" : ""}`, "aria-hidden": "true" },
    h("div", { class: "sk-line sk-w40" }),
    h("div", { class: "sk-line sk-w70" }),
    h("div", { class: "sk-line sk-w55" }),
    h("div", { class: "sk-line sk-block" }),
    h("div", { class: "sk-line sk-bar" }),
  );
}

function skeletonGrid(count = 6) {
  return h("div", { class: "round-grid" },
    Array.from({ length: count }, () => skeleton()),
  );
}

const ICON_EMPTY = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></svg>`;
const ICON_ERROR = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M12 8v5M12 16.5v.5"/><circle cx="12" cy="12" r="8.5"/></svg>`;

function notice({ title, body, icon = ICON_EMPTY, tone = "", action = null }) {
  return h("div", { class: `notice${tone ? ` notice-${tone}` : ""}`, role: "status" },
    h("span", { class: "notice-mark", "aria-hidden": "true", svg: icon }),
    h("h2", { class: "notice-title", text: title }),
    typeof body === "string" ? h("p", { class: "notice-body", text: body }) : body,
    action ? h("button", { class: "btn", type: "button", text: action.label, onclick: action.onClick }) : null,
  );
}

/** Turns any thrown error into copy that says what happened and what to do. */
function errorNotice(error, { context = "that", onRetry = null } = {}) {
  let title = "Something went wrong";
  let body = "The prediction service returned an unexpected response. Try again in a moment.";

  if (error instanceof ApiError && error.status === 0) {
    title = "Can't reach the prediction service";
    body = h("p", { class: "notice-body" },
      "Nothing answered at ",
      h("code", { text: API_BASE }),
      ". Check the API container is running and that it allows requests from this origin, then try again.",
    );
  } else if (error instanceof ApiError && error.status === 404) {
    title = `No prediction for ${context}`;
    body = error.detail || "The API has nothing on file for this one. Try a different selection.";
  } else if (error instanceof ApiError && error.status >= 500) {
    title = "The prediction service hit an error";
    body = error.detail || "The API responded with a server error. Try again in a moment.";
  } else if (error instanceof ApiError && error.detail) {
    body = error.detail;
  }

  return notice({
    title,
    body,
    icon: ICON_ERROR,
    tone: "error",
    action: onRetry ? { label: "Try again", onClick: onRetry } : null,
  });
}


/* ── shared team list (fetched once) ─────────────────────────────────── */

let teamsPromise = null;

function loadTeams() {
  if (!teamsPromise) {
    teamsPromise = api.teams().then((data) => {
      const teams = Array.isArray(data?.teams) ? data.teams.filter(Boolean) : [];
      if (!teams.length) throw new ApiError("empty team list", { status: 200 });
      return teams;
    }).catch((error) => {
      teamsPromise = null;   // let the retry button have another go
      throw error;
    });
  }
  return teamsPromise;
}

function fillTeamSelect(select, teams, selectedIndex = 0) {
  select.replaceChildren(
    ...teams.map((team) => h("option", { value: team, text: team })),
  );
  select.value = teams[Math.min(selectedIndex, teams.length - 1)];
}


/* ── view: round ─────────────────────────────────────────────────────── */

const roundOutput = $("#round-output");
const roundTally = $("#round-tally");
const mdSelect = $("#md-select");
const mdPrev = $("#md-prev");
const mdNext = $("#md-next");

let roundToken = 0;

function renderTally(predictions) {
  const counts = { home: 0, draw: 0, away: 0 };
  for (const match of predictions) counts[outcomeOf(match.prediction).key] += 1;

  const chips = [
    [predictions.length, predictions.length === 1 ? "match" : "matches"],
    [counts.home, counts.home === 1 ? "home win" : "home wins"],
    [counts.draw, counts.draw === 1 ? "draw" : "draws"],
    [counts.away, counts.away === 1 ? "away win" : "away wins"],
  ];

  fill(roundTally, chips.map(([count, label]) =>
    h("span", { class: "tally-chip" }, h("b", { text: String(count) }), ` ${label}`),
  ));
}

async function showRound(matchday) {
  const token = ++roundToken;
  roundTally.replaceChildren();
  fill(roundOutput, skeletonGrid());

  try {
    const data = await api.round(matchday);
    if (token !== roundToken) return;

    const predictions = Array.isArray(data?.predictions) ? data.predictions : [];
    if (!predictions.length) {
      fill(roundOutput, notice({
        title: `Matchday ${matchday} is empty`,
        body: "The API returned no fixtures for this round. Pick another matchday.",
      }));
      return;
    }

    renderTally(predictions);
    // No matchday box on these cards — the picker above already says which round.
    fill(roundOutput, h("div", { class: "round-grid" },
      predictions.map((match) => matchCard(match)),
    ));
  } catch (error) {
    if (token !== roundToken) return;
    fill(roundOutput, errorNotice(error, {
      context: `matchday ${matchday}`,
      onRetry: () => showRound(matchday),
    }));
  }
}

async function initRound() {
  mdSelect.replaceChildren(
    ...Array.from({ length: MATCHDAY_MAX - MATCHDAY_MIN + 1 }, (_, i) => {
      const value = String(MATCHDAY_MIN + i);
      return h("option", { value, text: value });
    }),
  );

  let startingMatchday = MATCHDAY_MIN;
  try {
    const data = await apiGet("/current-matchday");
    if (Number.isFinite(Number(data?.matchday))) {
      startingMatchday = Number(data.matchday);
    }
  } catch {
    // API didn't have an answer — just fall back to MATCHDAY_MIN, no need to
    // show an error for this, the round view's own error state will still
    // fire if that fallback matchday has nothing scheduled either.
  }

  mdSelect.value = String(startingMatchday);

  const syncSteppers = () => {
    const value = Number(mdSelect.value);
    mdPrev.disabled = value <= MATCHDAY_MIN;
    mdNext.disabled = value >= MATCHDAY_MAX;
  };

  const step = (delta) => {
    const value = Math.min(MATCHDAY_MAX, Math.max(MATCHDAY_MIN, Number(mdSelect.value) + delta));
    if (String(value) === mdSelect.value) return;
    mdSelect.value = String(value);
    syncSteppers();
    showRound(value);
  };

  mdSelect.addEventListener("change", () => {
    syncSteppers();
    showRound(Number(mdSelect.value));
  });
  mdPrev.addEventListener("click", () => step(-1));
  mdNext.addEventListener("click", () => step(1));

  syncSteppers();
  showRound(startingMatchday);
}


/* ── view: head to head ──────────────────────────────────────────────── */

const h2hHome = $("#h2h-home");
const h2hAway = $("#h2h-away");
const h2hSwap = $("#h2h-swap");
const h2hGo = $("#h2h-go");
const h2hHint = $("#h2h-hint");
const h2hOutput = $("#h2h-output");

let h2hReady = false;
let h2hToken = 0;

function validateH2H() {
  const clash = h2hHome.value && h2hHome.value === h2hAway.value;
  h2hGo.disabled = clash;
  h2hHint.textContent = clash ? "Pick two different teams." : "";
  return !clash;
}

async function predictH2H() {
  if (!validateH2H()) return;

  const home = h2hHome.value;
  const away = h2hAway.value;
  const token = ++h2hToken;
  fill(h2hOutput, skeleton({ featured: true }));

  try {
    const match = await api.game(home, away);
    if (token !== h2hToken) return;
    fill(h2hOutput, matchCard(match, { variant: "featured" }));
  } catch (error) {
    if (token !== h2hToken) return;
    fill(h2hOutput, errorNotice(error, {
      context: `${home} versus ${away}`,
      onRetry: predictH2H,
    }));
  }
}

async function initH2H() {
  if (h2hReady) return;
  h2hReady = true;

  fill(h2hOutput, skeleton({ featured: true }));

  let teams;
  try {
    teams = await loadTeams();
  } catch (error) {
    h2hReady = false;
    fill(h2hOutput, errorNotice(error, { context: "the team list", onRetry: initH2H }));
    return;
  }

  fillTeamSelect(h2hHome, teams, 0);
  fillTeamSelect(h2hAway, teams, 1);
  validateH2H();

  h2hHome.addEventListener("change", validateH2H);
  h2hAway.addEventListener("change", validateH2H);
  h2hGo.addEventListener("click", predictH2H);

  h2hSwap.addEventListener("click", () => {
    [h2hHome.value, h2hAway.value] = [h2hAway.value, h2hHome.value];
    h2hSwap.classList.toggle("is-turning");
    if (validateH2H() && h2hOutput.querySelector(".card")) predictH2H();
  });

  fill(h2hOutput, notice({
    title: "Pick two teams",
    body: "Any pairing works — the model will rate it whether or not the fixture is on the calendar.",
  }));
}


/* ── view: next match ────────────────────────────────────────────────── */

const nextTeam = $("#next-team");
const nextOutput = $("#next-output");

let nextReady = false;
let nextToken = 0;

async function showNextMatch() {
  const team = nextTeam.value;
  if (!team) return;

  const token = ++nextToken;
  fill(nextOutput, skeleton({ featured: true }));

  try {
    const match = await api.nextFor(team);
    if (token !== nextToken) return;
    fill(nextOutput, matchCard(match, { variant: "featured" }));
  } catch (error) {
    if (token !== nextToken) return;
    fill(nextOutput, errorNotice(error, {
      context: `${team}'s next match`,
      onRetry: showNextMatch,
    }));
  }
}

async function initNext() {
  if (nextReady) return;
  nextReady = true;

  fill(nextOutput, skeleton({ featured: true }));

  let teams;
  try {
    teams = await loadTeams();
  } catch (error) {
    nextReady = false;
    fill(nextOutput, errorNotice(error, { context: "the team list", onRetry: initNext }));
    return;
  }

  fillTeamSelect(nextTeam, teams, 0);
  nextTeam.addEventListener("change", showNextMatch);
  showNextMatch();
}


/* ── tabs ────────────────────────────────────────────────────────────── */

const tabs = [...document.querySelectorAll(".tab")];
const VIEW_INIT = { round: null, h2h: initH2H, next: initNext };

function activateTab(tab, { focus = false } = {}) {
  for (const candidate of tabs) {
    const active = candidate === tab;
    candidate.classList.toggle("is-active", active);
    candidate.setAttribute("aria-selected", String(active));
    candidate.tabIndex = active ? 0 : -1;
    $(`#${candidate.getAttribute("aria-controls")}`).hidden = !active;
  }
  if (focus) tab.focus();
  VIEW_INIT[tab.dataset.view]?.();
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => activateTab(tab));
  tab.addEventListener("keydown", (event) => {
    const offset = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: tabs.length - 1 - index }[event.key];
    if (offset === undefined) return;
    event.preventDefault();
    activateTab(tabs[(index + offset + tabs.length) % tabs.length], { focus: true });
  });
});


/* ── start ───────────────────────────────────────────────────────────── */
initRound();
