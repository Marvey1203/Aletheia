// src/main.js

import { invoke } from '@tauri-apps/api/core';

// Get references to our HTML elements.
// This is safe because the script is loaded at the end of the body.
const queryInput = document.querySelector('#query-input');
const reasonButton = document.querySelector('#reason-button');
const responseDiv = document.querySelector('#response-div');

// This function handles the logic for calling the backend
async function askAletheia() {
  const query = queryInput.value;
  if (!query) {
    responseDiv.innerText = 'Please enter a query.';
    return;
  }

  // Update the UI to show we are working
  responseDiv.innerText = 'Thinking...';
  reasonButton.disabled = true;
  queryInput.disabled = true;

  try {
    // Invoke the Rust command `invoke_backend` and pass the query
    const responseJson = await invoke('invoke_backend', { query });
    
    // The response from Python is a JSON string, so we parse it
    const trace = JSON.parse(responseJson);

    // Display the answer from the trace summary
    responseDiv.innerHTML = `
      <p><strong>Answer:</strong> ${trace.summary.answer}</p>
      <p><strong>Reasoning:</strong> ${trace.summary.reasoning}</p>
      <p><em>Trace ID: ${trace.trace_id}</em></p>
    `;

  } catch (error) {
    // If an error occurs, display it
    responseDiv.innerText = `Error: ${error}`;
    console.error(error);
  } finally {
    // Re-enable the UI elements
    reasonButton.disabled = false;
    queryInput.disabled = false;
  }
}

// Listen for clicks on the "Reason" button
reasonButton.addEventListener('click', askAletheia);

// Also allow pressing "Enter" in the input field
queryInput.addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    askAletheia();
  }
});