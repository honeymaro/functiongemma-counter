import "./style.css";
import { setupCounter, type CounterController } from "./counter.ts";
import {
  loadModel,
  generateFunctionCall,
  isModelLoaded,
  type FunctionCall,
} from "./functiongemma.ts";

let counterController: CounterController | null = null;

function executeFunction(functionCall: FunctionCall): string {
  if (!counterController) {
    return "Counter not initialized";
  }

  switch (functionCall.name) {
    case "increment":
      counterController.increment();
      return `Incremented counter to ${counterController.getValue()}`;

    case "decrement":
      counterController.decrement();
      return `Decremented counter to ${counterController.getValue()}`;

    case "set_counter": {
      const numStr = functionCall.args.number as string;
      const value = parseInt(numStr, 10);
      if (!isNaN(value)) {
        counterController.set_counter(value);
        return `Set counter to ${value}`;
      }
      return "Invalid value provided";
    }

    case "reset_counter":
      counterController.reset_counter();
      return "Counter reset to 0";

    default:
      return `Unknown function: ${functionCall.name}`;
  }
}

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
  <div class="container">
    <header>
      <h1>FunctionGemma Counter</h1>
      <p class="subtitle">Browser-based Function Calling with WebGPU</p>
      <p class="model-info">Model: <code>functiongemma-270m-it-ONNX</code></p>
    </header>

    <div class="counter-section">
      <div class="counter-display">
        <button id="counter" type="button"></button>
      </div>
    </div>

    <div class="model-section">
      <button id="load-model" type="button">Load Model</button>
      <div id="model-status" class="status">Model not loaded</div>
    </div>

    <div class="input-section">
      <input
        type="text"
        id="user-input"
        placeholder="Enter a command (e.g., 'increase', 'set to 10', 'reset')"
        disabled
      />
      <button id="send-btn" type="button" disabled>Send</button>
    </div>

    <div class="quick-commands">
      <h3>Quick Commands</h3>
      <div class="command-grid">
        <button class="quick-cmd" data-cmd="increase" disabled>+ Increment</button>
        <button class="quick-cmd" data-cmd="decrease" disabled>- Decrement</button>
        <button class="quick-cmd" data-cmd="set to 100" disabled>Set to 100</button>
        <button class="quick-cmd" data-cmd="reset" disabled>Reset</button>
      </div>
    </div>

    <div id="result" class="result"></div>

    <div class="examples-section">
      <div class="example-column">
        <h3>Increment Examples</h3>
        <ul class="example-list">
          <li><span class="example-cmd" data-cmd="increase the counter">increase the counter</span></li>
          <li><span class="example-cmd" data-cmd="add one">add one</span></li>
          <li><span class="example-cmd" data-cmd="plus one">plus one</span></li>
          <li><span class="example-cmd" data-cmd="+1">+1</span></li>
        </ul>
      </div>
      <div class="example-column">
        <h3>Decrement Examples</h3>
        <ul class="example-list">
          <li><span class="example-cmd" data-cmd="decrease the counter">decrease the counter</span></li>
          <li><span class="example-cmd" data-cmd="subtract one">subtract one</span></li>
          <li><span class="example-cmd" data-cmd="minus one">minus one</span></li>
          <li><span class="example-cmd" data-cmd="-1">-1</span></li>
        </ul>
      </div>
      <div class="example-column">
        <h3>Set Counter Examples</h3>
        <ul class="example-list">
          <li><span class="example-cmd" data-cmd="set to 42">set to 42</span></li>
          <li><span class="example-cmd" data-cmd="change value to 500">change value to 500</span></li>
          <li><span class="example-cmd" data-cmd="set counter to 99">set counter to 99</span></li>
        </ul>
      </div>
      <div class="example-column">
        <h3>Reset Examples</h3>
        <ul class="example-list">
          <li><span class="example-cmd" data-cmd="reset the counter">reset the counter</span></li>
          <li><span class="example-cmd" data-cmd="clear it">clear it</span></li>
          <li><span class="example-cmd" data-cmd="set to zero">set to zero</span></li>
        </ul>
      </div>
    </div>

    <div class="stats-section">
      <h3>Test Results</h3>
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-value">1730</div>
          <div class="stat-label">Test Cases</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">99.9%</div>
          <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">WebGPU</div>
          <div class="stat-label">Backend</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">270M</div>
          <div class="stat-label">Parameters</div>
        </div>
      </div>
    </div>

    <footer>
      <p>Built with <a href="https://huggingface.co/docs/transformers.js" target="_blank">Transformers.js</a> +
      <a href="https://huggingface.co/onnx-community/functiongemma-270m-it-ONNX" target="_blank">FunctionGemma</a></p>
    </footer>
  </div>
`;

counterController = setupCounter(
  document.querySelector<HTMLButtonElement>("#counter")!
);

const loadModelBtn = document.querySelector<HTMLButtonElement>("#load-model")!;
const modelStatus = document.querySelector<HTMLDivElement>("#model-status")!;
const userInput = document.querySelector<HTMLInputElement>("#user-input")!;
const sendBtn = document.querySelector<HTMLButtonElement>("#send-btn")!;
const resultDiv = document.querySelector<HTMLDivElement>("#result")!;
const quickCmds = document.querySelectorAll<HTMLButtonElement>(".quick-cmd");
const exampleCmds = document.querySelectorAll<HTMLSpanElement>(".example-cmd");

loadModelBtn.addEventListener("click", async () => {
  loadModelBtn.disabled = true;
  modelStatus.textContent = "Loading model...";
  modelStatus.className = "status loading";

  try {
    await loadModel((progress) => {
      if (progress.status === "loading") {
        const pct =
          progress.progress !== undefined
            ? ` (${Math.round(progress.progress)}%)`
            : "";
        modelStatus.textContent = `Loading ${progress.file}${pct}`;
      } else if (progress.status === "ready") {
        modelStatus.textContent = "Model ready!";
        modelStatus.className = "status ready";
      }
    });

    userInput.disabled = false;
    sendBtn.disabled = false;
    loadModelBtn.textContent = "✓ Model Loaded";
    loadModelBtn.classList.add("loaded");

    // Enable quick commands and example commands
    quickCmds.forEach((btn) => (btn.disabled = false));
    exampleCmds.forEach((span) => span.classList.add("clickable"));
  } catch (error) {
    modelStatus.textContent = `Error: ${error}`;
    modelStatus.className = "status error";
    loadModelBtn.disabled = false;
  }
});

async function handleUserInput(input?: string) {
  const command = input || userInput.value.trim();
  if (!command || !isModelLoaded()) return;

  sendBtn.disabled = true;
  userInput.disabled = true;
  quickCmds.forEach((btn) => (btn.disabled = true));
  resultDiv.innerHTML = `
    <div class="processing">
      <div class="spinner"></div>
      <span>Processing: "${command}"</span>
    </div>
  `;

  try {
    const startTime = performance.now();
    const functionCall = await generateFunctionCall(command);
    const elapsed = Math.round(performance.now() - startTime);

    if (functionCall) {
      const result = executeFunction(functionCall);
      resultDiv.innerHTML = `
        <div class="success">
          <div class="result-header">
            <span class="result-icon">✅</span>
            <span class="result-time">${elapsed}ms</span>
          </div>
          <div class="result-row">
            <span class="result-label">Input:</span>
            <span class="result-value">"${command}"</span>
          </div>
          <div class="result-row">
            <span class="result-label">Function:</span>
            <code class="result-value">${functionCall.name}(${Object.keys(functionCall.args).length > 0 ? JSON.stringify(functionCall.args) : ""})</code>
          </div>
          <div class="result-row">
            <span class="result-label">Result:</span>
            <span class="result-value">${result}</span>
          </div>
        </div>
      `;
    } else {
      resultDiv.innerHTML =
        '<div class="error">❌ Could not parse function call from model output</div>';
    }
  } catch (error) {
    resultDiv.innerHTML = `<div class="error">❌ Error: ${error}</div>`;
  }

  sendBtn.disabled = false;
  userInput.disabled = false;
  quickCmds.forEach((btn) => (btn.disabled = false));
  userInput.value = "";
  userInput.focus();
}

sendBtn.addEventListener("click", () => handleUserInput());
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    handleUserInput();
  }
});

// Quick command buttons
quickCmds.forEach((btn) => {
  btn.addEventListener("click", () => {
    const cmd = btn.dataset.cmd;
    if (cmd) handleUserInput(cmd);
  });
});

// Clickable examples
exampleCmds.forEach((span) => {
  span.addEventListener("click", () => {
    if (!isModelLoaded()) return;
    const cmd = span.dataset.cmd;
    if (cmd) handleUserInput(cmd);
  });
});
