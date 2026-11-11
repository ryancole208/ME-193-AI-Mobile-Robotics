// Polls /status once a second and keeps the MJPEG <img> alive across camera dropouts.
(() => {
  const POLL_MS = 1000;
  const RELOAD_MIN_GAP_MS = 3000;

  const $ = (id) => document.getElementById(id);
  const feed = $("feed");
  const offline = $("offline");

  let lastReload = Date.now();
  // Set whenever the stream may have died; cleared only by actually reopening it.
  let needReload = false;

  function showOffline(msg) {
    $("offline-msg").textContent = msg;
    offline.hidden = false;
    needReload = true;
  }

  // A cache-busting query makes the browser open a fresh stream connection.
  function reloadStream() {
    lastReload = Date.now();
    needReload = false;
    feed.src = "/video_feed?t=" + lastReload;
  }

  feed.addEventListener("error", () => showOffline("Camera disconnected — reconnecting…"));

  function setConnected(on) {
    $("dot").classList.toggle("on", on);
    $("conn").textContent = on ? "Connected" : "Disconnected";
  }

  async function poll() {
    let s;
    try {
      const r = await fetch("/status", { cache: "no-store" });
      s = await r.json();
    } catch {
      setConnected(false);
      showOffline("Server unreachable — retrying…");
      return;
    }

    setConnected(s.connected);
    $("index").textContent = s.camera_name ? `${s.camera_index ?? "-"} (${s.camera_name})` : s.camera_index;
    $("res").textContent = s.resolution ? `${s.resolution.width}×${s.resolution.height}` : "—";
    $("fps").textContent = s.connected ? s.fps.toFixed(1) : "—";

    if (!s.connected) {
      showOffline("Camera disconnected — reconnecting…");
    } else if (needReload) {
      // The server closes the stream on disconnect, so reopen it once the camera is back.
      // The gap stops a stream that errors instantly from being hammered every second.
      if (Date.now() - lastReload > RELOAD_MIN_GAP_MS) reloadStream();
    } else {
      offline.hidden = true;
    }
  }

  poll();
  setInterval(poll, POLL_MS);

  // ---- live finger count + lock demo (polled ~8x per second) ----
  const N_LEDS = 5;
  $("leds").replaceChildren(...Array.from({ length: N_LEDS }, () => {
    const d = document.createElement("span"); d.className = "led"; return d;
  }));
  let barsBuilt = false;

  function buildBars(classes) {
    $("bars").replaceChildren(...classes.flatMap((c) => {
      const name = document.createElement("span"); name.textContent = c;
      const track = document.createElement("div"); track.className = "track";
      const fill = document.createElement("div"); fill.className = "fill"; fill.id = "bar-" + c;
      track.appendChild(fill);
      const pct = document.createElement("span"); pct.className = "pct"; pct.id = "pct-" + c;
      return [name, track, pct];
    }));
    barsBuilt = true;
  }

  function renderLock(L) {
    const leds = $("leds").children;
    for (const led of leds) {
      led.className = "led" + (L.result === "unlocked" ? " ok" : L.result === "locked" ? " bad" : L.leds_on ? " on" : "");
    }
    $("digits").replaceChildren(...Array.from({ length: L.n_digits }, (_, i) => {
      const d = document.createElement("span");
      d.className = "digit" + (L.state === "awake" && i === L.entered.length ? " next" : "");
      d.textContent = L.entered[i] ?? "";
      return d;
    }));
    $("hold-bar").style.width = `${Math.round(L.progress * 100)}%`;
    $("lock-msg").textContent = L.message;
    const r = $("result");
    r.hidden = !L.result;
    if (L.result) {
      r.className = "result " + L.result;
      r.textContent = L.result === "unlocked" ? "UNLOCKED" : "STAYS LOCKED";
    }
  }

  async function pollLive() {
    let s;
    try { s = await (await fetch("/live", { cache: "no-store" })).json(); } catch { return; }
    $("pred-empty").hidden = s.model_loaded;
    $("pred").hidden = !s.model_loaded;
    $("press").disabled = !s.model_loaded;
    if (s.lock) renderLock(s.lock);
    if (!s.ready) { $("count").textContent = "–"; return; }
    const classes = Object.keys(s.probabilities);
    if (!barsBuilt) buildBars(classes);
    $("count").textContent = s.count;
    $("conf").textContent = `${Math.round(s.confidence * 100)}%`;
    const top = classes.reduce((a, b) => (s.probabilities[a] >= s.probabilities[b] ? a : b));
    for (const c of classes) {
      const v = s.probabilities[c];
      const fill = $("bar-" + c);
      fill.style.width = `${Math.round(v * 100)}%`;
      fill.classList.toggle("top", c === top);
      $("pct-" + c).textContent = `${Math.round(v * 100)}%`;
    }
    $("ims").textContent = s.inference_ms.toFixed(1);
    $("ifps").textContent = s.fps.toFixed(1);
    $("model").textContent = s.model;
  }

  $("press").addEventListener("click", () => fetch("/lock/press", { method: "POST" }));
  $("reset").addEventListener("click", () => fetch("/lock/reset", { method: "POST" }));
  // Keyboard: B = push button (keeps the hand under the camera free).
  document.addEventListener("keydown", (e) => {
    if (e.key === "b" || e.key === "B") fetch("/lock/press", { method: "POST" });
  });
  pollLive();
  setInterval(pollLive, 125);
})();
