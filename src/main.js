// src/main.js (V3 - Atlas Edition)

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

        queryInput.value = '';
        autoResizeTextarea(queryInput);
        setUIState(false, 'Sending...');

        appendMessage(query, 'user');
        const pendingBubble = appendMessage({ stage: 'Connecting', detail: 'Establishing link to Aletheia Core...' }, 'ai', true);
        
        try {
            const ack = await invoke('invoke_backend', { query, gearOverride });
            const { query_id } = JSON.parse(ack);

            const unlisten = await listen('telemetry-update', (event) => {
                const telemetry = JSON.parse(event.payload);

                if (telemetry.query_id === query_id) {
                    const bubbleContent = pendingBubble.querySelector('.message-content');
                    
                    if (telemetry.type === 'stage_update') {
                        // V3 UPGRADE: Handle the new, richer telemetry with memory context
                        let stageText = telemetry.stage.charAt(0).toUpperCase() + telemetry.stage.slice(1);
                        let detailText = '';

                        if (telemetry.stage === 'triage' && telemetry.payload.pathway) {
                            const pathway = telemetry.payload.pathway;
                            detailText = `Pathway selected: <strong>${pathway.execution_model}</strong>`;
                            
                            // Check for retrieved memories from the new context
                            const context = telemetry.payload.context;
                            if (context && context.long_term_memory && context.long_term_memory.length > 0) {
                                let memoryHtml = '<div class="retrieved-memories"><h4>Recalling Relevant Memories...</h4><ul>';
                                context.long_term_memory.forEach(mem => {
                                    memoryHtml += `<li>${mem.answer.substring(0, 80)}...</li>`;
                                });
                                memoryHtml += '</ul></div>';
                                pendingBubble.querySelector('.message-content').insertAdjacentHTML('afterbegin', memoryHtml);
                            }
                        } else if (telemetry.stage === 'enrich') {
                            detailText = `Refining plan with <strong>${telemetry.payload.final_plan.length}</strong> detailed steps...`;
                        } else if (telemetry.stage === 'execute') {
                            detailText = `Generating final answer with <strong>${telemetry.payload.pathway.execution_model}</strong>...`;
                        }
                        
                        bubbleContent.querySelector('.stage-indicator .stage').textContent = `${stageText}...`;
                        if(detailText) bubbleContent.querySelector('.stage-indicator .detail').innerHTML = detailText;

                    } else if (telemetry.type === 'final_result') {
                        const trace = telemetry.payload;
                        
                        const finalAnswerHtml = marked.parse(trace.summary.answer, {
                            highlight: (code, lang) => {
                                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                                return hljs.highlight(code, { language }).value;
                            }
                        });

                        const footer = `<footer class="message-footer">${trace.summary.reasoning}</footer>`;
                        // Replace the entire bubble content with the final answer and footer
                        pendingBubble.innerHTML = finalAnswerHtml + footer;
                        
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

    function appendMessage(content, sender, isPending = false) {
        const messageContainer = document.createElement('div');
        messageContainer.className = `message-container ${sender}`;

        const messageBubble = document.createElement('div');
        messageBubble.className = 'message-bubble';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (isPending) {
            contentDiv.innerHTML = `<div class="stage-indicator"><span class="stage">${content.stage}...</span><span class="detail">${content.detail}</span></div>`;
        } else {
            contentDiv.textContent = content;
        }
        
        messageBubble.appendChild(contentDiv);
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

    // Event Listeners
    reasonButton.addEventListener('click', askAletheia);
    queryInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            askAletheia();
        }
    });
    queryInput.addEventListener('input', () => autoResizeTextarea(queryInput));

    setUIState(true);
});