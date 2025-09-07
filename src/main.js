// src/main.js (V2 - Orchestra Edition)

import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { marked } from 'https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js'; 
import hljs from 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/es/highlight.min.js';

window.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const messageHistory = document.querySelector('#message-history');
    const queryInput = document.querySelector('#query-input');
    const reasonButton = document.querySelector('#reason-button');
    const gearSelector = document.querySelector('#gear-selector');

    // --- Core Application Logic ---

    async function askAletheia() {
        const query = queryInput.value.trim();
        if (!query) return;

        const selectedGearValue = gearSelector.querySelector('input[name="gear"]:checked').value;
        const gearOverride = selectedGearValue === 'auto' ? null : selectedGearValue;

        // --- UI State Management ---
        queryInput.value = '';
        autoResizeTextarea(queryInput);
        setUIState(false, 'Sending...');

        appendMessage(query, 'user');
        const pendingBubble = appendMessage('...', 'ai');
        
        try {
            const ack = await invoke('invoke_backend', { query, gearOverride });
            const { query_id } = JSON.parse(ack);

            const unlisten = await listen('telemetry-update', (event) => {
                const telemetry = JSON.parse(event.payload);

                if (telemetry.query_id === query_id) {
                    // --- Rich Telemetry Rendering ---
                    const bubbleContent = pendingBubble.querySelector('.message-content');
                    
                    if (telemetry.type === 'stage_update') {
                        // V2 UPGRADE: Display richer telemetry from the orchestra
                        let stageText = telemetry.stage.charAt(0).toUpperCase() + telemetry.stage.slice(1);
                        let detailText = '';

                        if (telemetry.stage === 'triage' && telemetry.payload.pathway) {
                            detailText = `Pathway selected: ${telemetry.payload.pathway.execution_model}`;
                        } else if (telemetry.stage === 'plan') {
                            detailText = `Sketching plan with ${telemetry.payload.initial_plan.length} steps...`;
                        } else if (telemetry.stage === 'enrich') {
                            detailText = `Refining plan with ${telemetry.payload.final_plan.length} detailed steps...`;
                        } else if (telemetry.stage === 'execute') {
                            detailText = 'Generating final answer...';
                        }
                        
                        bubbleContent.innerHTML = `<span class="stage">${stageText}...</span><span class="detail">${detailText}</span>`;

                    } else if (telemetry.type === 'final_result') {
                        const trace = telemetry.payload;
                        
                        const finalAnswerHtml = marked.parse(trace.summary.answer, {
                            highlight: (code, lang) => {
                                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                                return hljs.highlight(code, { language }).value;
                            }
                        });

                        // Add a footer to the message with metadata
                        const footer = `<footer class="message-footer">${trace.summary.reasoning}</footer>`;
                        bubbleContent.innerHTML = finalAnswerHtml + footer;
                        
                        unlisten();
                        setUIState(true);
                    }
                }
            });
        } catch (error) {
            const errorMessage = `Error: ${error.toString()}`;
            const bubbleContent = pendingBubble.querySelector('.message-content');
            bubbleContent.innerHTML = `<span style="color: #ff8a8a;">${errorMessage}</span>`;
            console.error(errorMessage);
            setUIState(true);
        }
    }

    function appendMessage(text, sender) {
        const messageContainer = document.createElement('div');
        messageContainer.className = `message-container ${sender}`;

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.textContent = text;
        messageBubble.appendChild(content);

        messageContainer.appendChild(messageBubble);
        messageHistory.appendChild(messageContainer);

        messageHistory.scrollTop = messageHistory.scrollHeight;
        return messageBubble;
    }

    function setUIState(enabled, buttonText = '➤') {
        queryInput.disabled = !enabled;
        reasonButton.disabled = !enabled;
        reasonButton.textContent = buttonText;
        if (enabled) {
            queryInput.focus();
        }
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

    // Initial state
    setUIState(true);
});