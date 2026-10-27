const timelineData = {
  s1:  ["F1","F2","F3","F4","·","·","·","B1","F5","B2","F6","B3","F7"],
  s2a: ["·","F1","·","F3","·","B1","F5","·","B3","·","F7","·","B5"],
  s2b: ["·","·","F2","·","F4","·","B2","F6","·","B4","F8","·","B6"],
  s3:  ["·","·","F1","B1","F2","B2","F3","B3","F4","B4","F5","B5","F6"]
};

const nodePositions = {
  input: [10, 50], s1: [26.25, 50], s2a: [62.25, 27], s2b: [62.25, 72], s3: [88.25, 50], loss: [97, 50]
};

const frames = [
  { kind:"ready", kicker:"PIPELINE PRIMED", text:"Press play. Each tick is one scheduling decision; several GPUs can work during the same tick.", work:{}, particles:[{id:1,node:"input",dir:"f"},{id:2,node:"input",dir:"f"},{id:3,node:"input",dir:"f"},{id:4,node:"input",dir:"f"}] },
  { kind:"forward", kicker:"m1 ENTERS · FORWARD", text:"GPU 0 reads its latest weight and sends m1's activation toward Stage 2.", work:{s1:"F · m1"}, particles:[{id:1,node:"s1",dir:"f"},{id:2,node:"input",dir:"f"},{id:3,node:"input",dir:"f"},{id:4,node:"input",dir:"f"}] },
  { kind:"forward", kicker:"PIPELINE FILLING", text:"m1 routes to Replica A while GPU 0 immediately starts m2. Different microbatches now overlap.", work:{s1:"F · m2",s2a:"F · m1"}, particles:[{id:1,node:"s2a",dir:"f"},{id:2,node:"s1",dir:"f"},{id:3,node:"input",dir:"f"},{id:4,node:"input",dir:"f"}] },
  { kind:"forward", kicker:"m3 STASHES W⁰", text:"Follow m3: Stage 1 computes its activation with W¹=v0 and saves that exact weight version for backward.", work:{s1:"F · m3",s2b:"F · m2",s3:"F · m1"}, particles:[{id:1,node:"s3",dir:"f"},{id:2,node:"s2b",dir:"f"},{id:3,node:"s1",dir:"f"},{id:4,node:"input",dir:"f"}] },
  { kind:"backward", kicker:"m1 TURNS AROUND", text:"At the loss, m1 becomes a backward gradient. It now travels left while later activations still travel right.", work:{s1:"F · m4",s3:"B · m1"}, particles:[{id:1,node:"loss",dir:"b"},{id:2,node:"s3",dir:"f"},{id:3,node:"s2a",dir:"f"},{id:4,node:"s1",dir:"f"}] },
  { kind:"backward", kicker:"TWO DIRECTIONS, ONE CLOCK", text:"m1's gradient returns through Replica A—the same replica that ran its forward pass. m2 reaches the loss.", work:{s2a:"B · m1",s3:"F · m2"}, particles:[{id:1,node:"s2a",dir:"b"},{id:2,node:"loss",dir:"b"},{id:3,node:"s3",dir:"f"},{id:4,node:"s2b",dir:"f"}] },
  { kind:"backward", kicker:"LIVE W¹ ADVANCES", text:"m1 updates Stage 1's live weight from v0 → v1. m3's stashed v0 is kept intact.", work:{s1:"B · m1",s3:"B · m2"}, particles:[{id:1,node:"s1",dir:"b"},{id:2,node:"s3",dir:"b"},{id:3,node:"loss",dir:"b"},{id:4,node:"s3",dir:"f"}], weights:{s1:1} },
  { kind:"backward", kicker:"m3 TURNS AROUND", text:"m3's loss is known. Its gradient begins the return trip as m2 runs backward on Replica B.", work:{s2b:"B · m2",s3:"B · m3"}, particles:[{id:2,node:"s2b",dir:"b"},{id:3,node:"s3",dir:"b"},{id:4,node:"loss",dir:"b"}], weights:{s1:1,s2a:3} },
  { kind:"backward", kicker:"m2 UPDATES LIVE W¹", text:"Stage 1 advances again, v1 → v2. The live weight is now newer than the v0 copy held for m3.", work:{s1:"B · m2",s2a:"B · m3",s3:"F · m4"}, particles:[{id:2,node:"s1",dir:"b"},{id:3,node:"s2a",dir:"b"},{id:4,node:"s3",dir:"f"}], weights:{s1:2,s2a:4,s2b:3} },
  { kind:"backward", kicker:"m3 READS ITS STASH", text:"On Stage 2, m3's gradient is computed with the weight version its forward pass used—not necessarily the newest live copy.", work:{s1:"F · m5",s2a:"B · m3",s3:"B · m4"}, particles:[{id:3,node:"s2a",dir:"b"},{id:4,node:"s3",dir:"b"}], weights:{s1:2,s2a:4,s2b:3,s3:4} },
  { kind:"backward", kicker:"THE STALE-WEIGHT MOMENT", text:"m3 reaches Stage 1. Compute g₃ with stashed v0; apply that stale gradient to current live v2. The model never rewinds.", work:{s1:"B · m3",s2b:"B · m4"}, particles:[{id:3,node:"s1",dir:"b"},{id:4,node:"s2b",dir:"b"}], weights:{s1:3,s2a:4,s2b:4,s3:4} },
  { kind:"backward", kicker:"UPDATE COMMITTED", text:"Live W¹ is now v3 = v2 − η·g₃(v0). Once m3's backward pass completes, its v0 stash can be released.", work:{s1:"COMMIT v3"}, particles:[{id:3,node:"input",dir:"b"},{id:4,node:"s2b",dir:"b"}], weights:{s1:3,s2a:4,s2b:4,s3:4} },
  { kind:"ready", kicker:"STEADY STATE", text:"The 1F1B rhythm continues: each stage alternates forward and backward work while weight versions protect the math.", work:{s1:"F · m7",s2a:"B · m5",s2b:"F · m8",s3:"B · m5"}, particles:[], weights:{s1:3,s2a:4,s2b:4,s3:4} }
];

let current = 0;
let timer = null;
let delay = 1100;

const els = {
  play: document.querySelector("#play-btn"), step: document.querySelector("#step-btn"), reset: document.querySelector("#reset-btn"),
  stepLabel: document.querySelector("#step-label"), kind: document.querySelector("#event-kind"), arrow: document.querySelector("#event-arrow"),
  direction: document.querySelector("#event-direction"), kicker: document.querySelector("#event-kicker"), text: document.querySelector("#event-text"), particles: document.querySelector("#particles")
};

function buildTimeline() {
  const timeline = document.querySelector("#timeline");
  const rows = [["GPU 0",timelineData.s1],["GPU 1",timelineData.s2a],["GPU 2",timelineData.s2b],["GPU 3",timelineData.s3]];
  timeline.innerHTML = `<span></span>${Array.from({length:13},(_,i)=>`<span class="track-cell tick" data-tick="${i}">${String(i).padStart(2,"0")}</span>`).join("")}`;
  rows.forEach(([label,data]) => {
    timeline.insertAdjacentHTML("beforeend", `<span class="track-label">${label}</span>${data.map((v,i)=>`<span class="track-cell ${v[0]==="F"?"forward":v[0]==="B"?"backward":""}" data-time="${i}">${v}</span>`).join("")}`);
  });
}

function renderParticles(frame) {
  const liveIds = new Set(frame.particles.map(p=>p.id));
  els.particles.querySelectorAll(".particle").forEach(el => { if(!liveIds.has(Number(el.dataset.id))) el.remove(); });
  frame.particles.forEach(p => {
    let el = els.particles.querySelector(`[data-id="${p.id}"]`);
    if (!el) {
      el = document.createElement("div"); el.className = `particle m${p.id}`; el.dataset.id = p.id; el.textContent = p.dir === "b" ? `g${p.id}` : `m${p.id}`; els.particles.appendChild(el);
    }
    const [x,y] = nodePositions[p.node];
    el.style.left = `${x}%`; el.style.top = `${y}%`;
    el.textContent = p.dir === "b" ? `g${p.id}` : `m${p.id}`;
    el.classList.toggle("backward", p.dir === "b"); el.classList.toggle("focus", p.id === 3);
  });
}

function render(index) {
  current = index;
  const frame = frames[index];
  els.stepLabel.textContent = `${String(index).padStart(2,"0")} / 12`;
  els.kicker.textContent = frame.kicker; els.text.textContent = frame.text;
  const backward = frame.kind === "backward";
  els.kind.textContent = frame.kind === "ready" ? "READY" : backward ? "GRADIENT" : "ACTIVATION";
  els.arrow.textContent = backward ? "←" : "→";
  els.direction.classList.toggle("backward-mode", backward);
  document.querySelectorAll(".gpu-card").forEach(card => {
    const key = card.dataset.node; const task = frame.work[key];
    card.classList.remove("active-forward","active-backward");
    if (task) card.classList.add(task.startsWith("B") ? "active-backward" : "active-forward");
    document.querySelector(`#slot-${key}`).textContent = task || "IDLE";
  });
  const weights = {
    s1: index >= 10 ? 3 : index >= 8 ? 2 : index >= 6 ? 1 : 0,
    s2a: index >= 9 ? 2 : index >= 5 ? 1 : 0,
    s2b: index >= 10 ? 2 : index >= 7 ? 1 : 0,
    s3: index >= 9 ? 4 : index >= 7 ? 3 : index >= 6 ? 2 : index >= 4 ? 1 : 0
  };
  document.querySelector("#w-s1").textContent = `W¹ · v${weights.s1}`;
  document.querySelector("#w-s2a").textContent = `W² · v${weights.s2a}`;
  document.querySelector("#w-s2b").textContent = `W² · v${weights.s2b}`;
  document.querySelector("#w-s3").textContent = `W³ · v${weights.s3}`;
  document.querySelectorAll(".track-cell").forEach(c => c.classList.toggle("current", Number(c.dataset.time ?? c.dataset.tick) === index));
  renderParticles(frame);
}

function stop() {
  clearInterval(timer); timer = null; els.play.classList.remove("playing"); els.play.querySelector(".play-copy").textContent = current === frames.length-1 ? "REPLAY" : "PLAY TRACE"; els.play.querySelector(".play-icon").textContent = "▶";
}
function play() {
  if (timer) { stop(); return; }
  if (current === frames.length-1) render(0);
  els.play.classList.add("playing"); els.play.querySelector(".play-copy").textContent = "PAUSE"; els.play.querySelector(".play-icon").textContent = "Ⅱ";
  timer = setInterval(() => { if (current >= frames.length-1) stop(); else render(current+1); }, delay);
}

els.play.addEventListener("click", play);
els.step.addEventListener("click", () => { stop(); render(current >= frames.length-1 ? 0 : current+1); });
els.reset.addEventListener("click", () => { stop(); render(0); });
document.querySelector("#speed-range").addEventListener("input", e => {
  delay = 2300 - Number(e.target.value); document.querySelector("#speed-label").textContent = `${(1100/delay).toFixed(1).replace(".0","")}×`;
  if (timer) { stop(); play(); }
});

document.querySelectorAll(".stash-toggle button").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll(".stash-toggle button").forEach(b => b.classList.toggle("active", b === button));
  const naive = button.dataset.mode === "naive";
  document.querySelector("#weight-story").classList.toggle("naive", naive);
  document.querySelector("#backward-title").textContent = naive ? "Compute with the live weight." : "Compute with the stash.";
  document.querySelector("#context-weight").textContent = naive ? "W²" : "W⁰";
  document.querySelector(".gradient-token strong").textContent = naive ? "g₃(W²)" : "g₃(W⁰)";
  document.querySelector("#equation-card strong").textContent = naive ? "W³ = W² − η · g₃(W²)" : "W³ = W² − η · g₃(W⁰)";
  document.querySelector("#backward-explanation").textContent = naive ? "But m3's activations came from W⁰. Mixing them with W² breaks the forward/backward chain." : "The derivative is internally consistent with m3’s forward pass, even though it is stale relative to W².";
  document.querySelector("#answer-copy").innerHTML = naive ? "<b>This is the dangerous mismatch.</b> Backward uses W² with activations produced by W⁰, so the result is not the gradient of a consistent forward computation." : "<b>W⁰ answers “how was this gradient computed?”</b> W² answers “where should we apply it now?” Stashed weights are read-only context—not an update target.";
}));

buildTimeline(); render(0);
