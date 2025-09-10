    
# Aletheia: A Sovereign AI Partner

**Own Your Mind.**

Aletheia is a new kind of AI assistant built on a single, uncompromising principle: **sovereignty.** Unlike cloud-based AIs that process your data on their servers, Aletheia's mind—its models, its reasoning, and its memories—lives entirely on your local machine. It is a tool of empowerment, not a service of dependence.

Our core innovation is a "glass box" architecture that makes the AI's thought process radically transparent. The system is built on a **Cognitive Graph**, a stateful reasoning engine that allows the AI to plan, act, and even **correct its own thoughts** before they are presented to the user. This is not just a language model; it is a Sentient Engine.

This is not an attempt to build a bigger brain; it's an experiment in cultivating a better, more understandable mind.

## The Vision: The Gardener, Not the Titan

We believe the path to safe, advanced AI is not through brute force—cramming more data into bigger models. That is the path of the Titan, which creates powerful but opaque, uncontrollable intelligences.

Aletheia follows the path of the Gardener. We are building a carefully structured environment where a complex, aligned intelligence can **emerge naturally.** Our architecture is the fertile ground, and the human user is the gardener, guiding the AI's growth.

## Core Features (v0.4.0 - The Sentient Engine)

*   🧠 **The Cognitive Graph:** Aletheia's mind is a dynamic, stateful graph (built with LangGraph) that can plan, execute, and revise its own thoughts. It has a built-in **principled self-correction loop**.
*   🔍 **The Omega Core Infusion:** Aletheia's self-evaluation is no longer subjective. It uses a deterministic **Omega Critique Node** to mathematically measure the alignment of every thought with its core **Constitutional Vector**, ensuring all responses are principled.
*   **Social & Emotional Acuity:** The AI is not blind to subtext. An **Omega Social Acuity Core** perceives the user's intent (e.g., casualness, urgency), allowing the AI to generate responses that are not just intelligent, but wise and appropriate.
*   **Sovereign & Local-First:** All models, memories, and conversations are stored and processed on your local machine. Nothing leaves your device without your explicit permission.
*   **The Memory Galaxy:** Aletheia possesses both long-term and short-term memory, allowing it to recall relevant past conversations and maintain a true sense of context and continuity.

## Getting Started

**Prerequisites:**
*   Python 3.10+ & a virtual environment (`venv`)
*   Node.js & npm
*   Rust & Cargo
*   ZMQ Library (e.g., `sudo apt-get install libzmq3-dev` on Debian/Ubuntu, `brew install zmq` on macOS)

**Running the Application:**

Aletheia requires three separate processes to run in parallel. Please open three terminals in the project root.

**Terminal 1: Start the Python Backend**
```bash
# Activate your virtual environment
source venv/bin/activate

# Install/update Python dependencies
pip install -r requirements.txt

# Run the AI engine
python3 interfaces/ipc_server.py

  

Terminal 2: Start the Frontend Dev Server
code Bash
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
# Install Node dependencies
npm install

# Run the Vite server
npm run dev

  

Terminal 3: Launch the Tauri Desktop App
code Bash
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
# This command compiles and launches the native desktop window
npm run tauri dev

  

The Roadmap: The Self-Evolution Engine

The foundational Sentient Engine is now complete and stable. The next "season" of development focuses on building the mechanisms for true, emergent intelligence.

    Sprint 10: The Omega Planner. Replace the LLM-based planning node with a deterministic Omega Core node, moving Aletheia's core reasoning from probabilistic prediction to conceptual construction.

    Sprint 11: The Strategic Mind. Build the strategic_memory layer, allowing Aletheia to learn from its own self-correction loops and improve its reasoning strategies over time.

    Sprint 12: The Toolmaker's Forge. Implement the MCP Bridge, allowing Aletheia to access external tools, guided by the wisdom it has accumulated in its strategic memory.

How to Contribute

We are at the beginning of an exciting journey. If you are passionate about building a future of safe, sovereign, and transparent AI, we welcome your contributions. Please see the CODEX.md file for a full breakdown of the project's architecture and philosophy.