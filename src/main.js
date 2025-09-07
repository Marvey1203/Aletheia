// src/main.js

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { marked } from 'https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js'; 
import hljs from 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/es/highlight.min.js';

// This listener ensures our script only runs after the HTML is fully loaded.
window.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const messageHistory = document.querySelector('#message-history');
    const queryInput = document.querySelector('#query-input');
    const reasonButton = document.querySelector('#reason-button');
    const gearSelector = document.querySelector('#gear-selector'); // This will now be found.

    // --- Core Application Logic ---

    async function askAletheia() {
        const query = queryInput.value.trim();
        if (!query) return;

        const selectedGear = gearSelector.querySelector('input[name="gear"]:checked').value;
        const gearOverride = selectedGear === 'auto' ? null : selectedGear;

        queryInput.value = '';
        queryInput.disabled = true;
        reasonButton.disabled = true;
        autoResizeTextarea(queryInput);

        appendMessage(query, 'user');

        const pendingBubble = appendMessage('...', 'ai');
        
        try {
            const ack = await invoke('invoke_backend', { query, gearOverride });
            const { query_id } = JSON.parse(ack);

            const unlisten = await listen('telemetry-update', (event) => {
                const telemetry = JSON.parse(event.payload);

                if (telemetry.query_id === query_id) {
                    const bubbleContent = pendingBubble.querySelector('.message-content');

                    if (telemetry.type === 'stage_update') {
                        bubbleContent.textContent = `${telemetry.stage.charAt(0).toUpperCase() + telemetry.stage.slice(1)}...`;
                    } else if (telemetry.type === 'final_result') {
                        const trace = telemetry.payload;
                        
                        const finalAnswerHtml = marked.parse(trace.summary.answer, {
                            highlight: (code, lang) => {
                                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                                return hljs.highlight(code, { language }).value;
                            }
                        });

                        bubbleContent.innerHTML = finalAnswerHtml;
                        
                        unlisten();
                        enableUI();
                    }
                }
            });
        } catch (error) {
            const errorMessage = `Error: ${error.toString()}`;
            const bubbleContent = pendingBubble.querySelector('.message-content');
            bubbleContent.innerHTML = `<span style="color: #ff8a8a;">${errorMessage}</span>`;
            console.error(errorMessage);
            enableUI();
        }
    }

    function appendMessage(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.className = `message-container ${sender}`;

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        // Use a dedicated div for content to make updates easier
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        messageBubble.appendChild(content);

        messageContainer.appendChild(messageBubble);
        messageHistory.appendChild(messageContainer);

        messageHistory.scrollTop = messageHistory.scrollHeight;
        return messageBubble;
    }

    function enableUI() {
        queryInput.disabled = false;
        reasonButton.disabled = false;
        queryInput.focus();
    }

    function autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = `${textarea.scrollHeight}px`;
    }

    // --- Event Listeners ---
    reasonButton.addEventListener('click', askAletheia);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            askAletheia();
        }
    });
    queryInput.addEventListener('input', () => autoResizeTextarea(queryInput));

    // Initial focus
    queryInput.focus();
});