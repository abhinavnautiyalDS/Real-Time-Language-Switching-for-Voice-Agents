const API_BASE = (window.CLASSIFIER_API_BASE || "").replace(/\/$/, "");
const input = document.getElementById("sentence");
const button = document.getElementById("classify-button");
const errorBox = document.getElementById("error-message");
const statusText = document.getElementById("api-status");
const statusDot = document.getElementById("status-dot");

input.addEventListener("input", () => {
  document.getElementById("char-count").textContent = input.value.length + " / 2000";
});
document.querySelectorAll("[data-sample]").forEach((item) => item.addEventListener("click", () => {
  input.value = item.dataset.sample;
  input.dispatchEvent(new Event("input"));
  input.focus();
}));

async function checkHealth() {
  if (!API_BASE) { statusText.textContent = "API endpoint not configured"; return; }
  try {
    const response = await fetch(API_BASE + "/api/health");
    const data = await response.json();
    if (!response.ok || !data.model_loaded) throw new Error("Model unavailable");
    statusText.textContent = "API connected · model ready";
    statusDot.style.background = "#eee";
  } catch (_) { statusText.textContent = "API unavailable"; }
}
function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }

function render(data) {
  document.getElementById("empty-result").hidden = true;
  document.getElementById("prediction").hidden = false;
  document.getElementById("result-state").textContent = "INFERENCE COMPLETE";
  document.getElementById("predicted-language").textContent = data.prediction.language;
  document.getElementById("predicted-code").textContent = data.prediction.code;
  document.getElementById("confidence").textContent = data.prediction.score.toFixed(2) + "%";
  document.getElementById("latency").textContent = data.inference_ms.toFixed(2) + " ms";
  document.getElementById("result-note").textContent = data.note || "";
  const target = document.getElementById("alternatives");
  target.replaceChildren();
  data.alternatives.forEach((item) => {
    const row = document.createElement("div"); row.className = "alternative";
    const name = document.createElement("span"); name.textContent = item.language;
    const track = document.createElement("div"); track.className = "bar-track";
    const fill = document.createElement("div"); fill.className = "bar-fill";
    fill.style.width = Math.max(0, Math.min(100, item.score)) + "%"; track.append(fill);
    const value = document.createElement("span"); value.className = "alternative-value"; value.textContent = item.score.toFixed(1) + "%";
    row.append(name, track, value); target.append(row);
  });
}
button.addEventListener("click", async () => {
  errorBox.hidden = true;
  const text = input.value.trim();
  if (!text) { showError("Enter a sentence before running the classifier."); input.focus(); return; }
  if (!API_BASE) { showError("Set CLASSIFIER_API_BASE in config.js to the public API URL."); return; }
  button.disabled = true; button.textContent = "Classifying…";
  document.getElementById("result-state").textContent = "RUNNING INFERENCE";
  try {
    const response = await fetch(API_BASE + "/api/classify", {
      method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Classifier request failed.");
    render(data);
    statusText.textContent = "API connected · model ready";
    statusDot.style.background = "#eee";
  } catch (error) {
    document.getElementById("result-state").textContent = "REQUEST FAILED";
    showError(error.message || "Could not reach classifier API.");
  } finally { button.disabled = false; button.textContent = "↗   Classify language"; }
});
checkHealth();
