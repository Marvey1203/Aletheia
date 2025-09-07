// This is the final, correct version.

window.addEventListener("DOMContentLoaded", () => {
  // This is the only line that has changed.
  // We are now using the full, correct path to the invoke function.
  const invoke = window.__TAURI__.core.invoke;

  const queryInput = document.querySelector("#query-input");
  const reasonButton = document.querySelector("#reason-button");
  const responseContainer = document.querySelector("#response-container");

  reasonButton.addEventListener("click", async () => {
    const query = queryInput.value;
    if (!query) return;

    responseContainer.textContent = "Aletheia is thinking...";
    try {
      const trace = await invoke("reason", { query });
      responseContainer.textContent = JSON.stringify(trace.summary, null, 2);
    } catch (error) {
      responseContainer.textContent = `Error: ${error}`;
    }
  });
});