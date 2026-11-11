// Data-collection page: pick a class, run a burst (countdown -> server captures), see counts.
(() => {
  const CFG = window.PAGE_CFG;
  const $ = (id) => document.getElementById(id);
  const pad = (n) => String(n).padStart(2, "0");

  let selected = null;
  let busy = false;

  // ---- session ID: train1 / train2 / val1 (remembered between page loads) ----
  try { $("session").value = localStorage.getItem("session") || "train1"; } catch { $("session").value = "train1"; }
  $("session").addEventListener("change", () => { try { localStorage.setItem("session", $("session").value.trim()); } catch {} });
  $("seconds").value = CFG.burst_seconds;
  $("fps").value = CFG.burst_fps;
  $("target").textContent = CFG.target;

  // ---- class buttons ----
  CFG.classes.forEach((c, i) => {
    const b = document.createElement("button");
    b.className = "btn cls";
    b.dataset.cls = c;
    b.innerHTML = `<span class="key">${CFG.counts[c]}</span>${CFG.display[c]}`;
    b.addEventListener("click", () => select(c));
    $("classes").appendChild(b);
  });

  function select(c) {
    selected = c;
    document.querySelectorAll(".cls").forEach((b) => b.classList.toggle("sel", b.dataset.cls === c));
    $("howto").textContent = CFG.howto[c];
  }

  // ---- beeps (Web Audio; no files needed) ----
  let audio;
  function beep(freq, ms) {
    audio = audio || new AudioContext();
    const o = audio.createOscillator();
    const g = audio.createGain();
    o.frequency.value = freq;
    g.gain.value = 0.15;
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + ms / 1000);
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function setMsg(text, bad = false) {
    $("msg").textContent = text;
    $("msg").classList.toggle("bad", bad);
  }

  function setBusy(on) {
    busy = on;
    $("burst").disabled = on;
    $("undo").disabled = on || !canUndo;
    document.querySelectorAll(".cls").forEach((b) => (b.disabled = on));
  }

  // ---- burst ----
  async function runBurst() {
    if (busy) return;
    if (!selected) return setMsg("Pick a class first.", true);
    const session = $("session").value.trim();
    if (!/^[A-Za-z0-9_-]{1,48}$/.test(session)) return setMsg("Session ID: letters, digits, _ and - only.", true);

    setBusy(true);
    $("bar").style.width = "0%";
    const cd = $("countdown");
    cd.hidden = false;
    for (let n = CFG.countdown; n > 0; n--) {
      cd.textContent = n;
      beep(600, 80);
      await sleep(1000);
    }
    cd.textContent = "●";
    beep(1000, 200);

    let r;
    try {
      r = await fetch("/collect/burst", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ class_id: selected, session_id: session,
                               seconds: parseFloat($("seconds").value), fps: parseFloat($("fps").value) }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.message);
    } catch (e) {
      cd.hidden = true;
      setMsg(`Burst failed: ${e.message}`, true);
      setBusy(false);
      return;
    }

    // Poll until the server-side burst ends.
    let s;
    for (;;) {
      await sleep(150);
      s = await (await fetch("/collect/status", { cache: "no-store" })).json();
      const pct = s.expected ? (100 * s.captured) / s.expected : 0;
      $("bar").style.width = `${pct}%`;
      setMsg(`Capturing ${CFG.display[selected]}: ${s.captured}/${s.expected}`);
      if (!s.active) break;
    }
    cd.hidden = true;
    beep(800, 120);
    await sleep(160);
    beep(500, 200);
    if (s.error) setMsg(`Burst stopped: ${s.error} (saved ${s.captured})`, true);
    else setMsg(`Saved ${s.captured} frames of ${CFG.display[selected]} to session ${session}.`);
    canUndo = s.can_undo;
    setBusy(false);
    refreshCounts();
  }

  let canUndo = false;
  async function undo() {
    if (busy) return;
    if (!confirm("Delete the frames of the most recent burst?")) return;
    const r = await fetch("/collect/undo", { method: "POST" });
    const j = await r.json();
    setMsg(j.ok ? `Undone: ${j.message}` : j.message, !j.ok);
    const s = await (await fetch("/collect/status", { cache: "no-store" })).json();
    canUndo = s.can_undo;
    $("undo").disabled = !canUndo;
    refreshCounts();
  }

  $("burst").addEventListener("click", runBurst);
  $("undo").addEventListener("click", undo);

  // Keyboard: 0-5 pick that finger count, Space starts a burst (one hand stays under the camera).
  const byCount = Object.fromEntries(CFG.classes.map((c) => [CFG.counts[c], c]));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    const i = parseInt(e.key, 10);
    if (i in byCount && !busy) select(byCount[i]);
    else if (e.code === "Space") { e.preventDefault(); runBurst(); }
  });

  // ---- counts table ----
  async function refreshCounts() {
    let c;
    try { c = await (await fetch("/collect/counts", { cache: "no-store" })).json(); } catch { return; }
    const t = $("counts");
    const head = `<tr><th>Class</th>${c.sessions.map((s) => `<th>${s}</th>`).join("")}<th>Total</th></tr>`;
    const sessTot = Object.fromEntries(c.sessions.map((s) => [s, 0]));
    let grand = 0;
    const rows = c.classes.map((cls) => {
      const cells = c.sessions.map((s) => {
        const n = c.counts[cls][s] || 0;
        sessTot[s] += n;
        return `<td>${n || ""}</td>`;
      });
      const tot = Object.values(c.counts[cls]).reduce((a, b) => a + b, 0);
      grand += tot;
      return `<tr class="${tot < c.target ? "low" : ""}"><td>${CFG.display[cls]}</td>${cells.join("")}<td class="tot">${tot}</td></tr>`;
    });
    const foot = `<tr class="foot"><td>Session total</td>${c.sessions.map((s) => `<td>${sessTot[s]}</td>`).join("")}<td class="tot">${grand}</td></tr>`;
    t.innerHTML = head + rows.join("") + foot;
  }

  // ---- keep the stream alive like the main page ----
  const feed = $("feed");
  let needReload = false;
  let lastReload = Date.now();
  feed.addEventListener("error", () => { $("offline").hidden = false; needReload = true; });
  setInterval(async () => {
    let s;
    try { s = await (await fetch("/status", { cache: "no-store" })).json(); }
    catch { $("offline").hidden = false; needReload = true; return; }
    if (!s.connected) { $("offline").hidden = false; needReload = true; }
    else if (needReload && Date.now() - lastReload > 3000) {
      lastReload = Date.now();
      needReload = false;
      feed.src = "/video_feed?overlay=collect&t=" + lastReload;
    } else if (!needReload) $("offline").hidden = true;
  }, 1000);

  select(CFG.classes[0]);
  refreshCounts();
  fetch("/collect/status").then((r) => r.json()).then((s) => { canUndo = s.can_undo; $("undo").disabled = !canUndo; });
})();
