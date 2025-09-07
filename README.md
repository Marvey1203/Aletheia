IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END

    
# Aletheia: A Sovereign AI Partner

**Own Your Mind.**

Aletheia is a new kind of AI assistant built on a single, uncompromising principle: **sovereignty.** Unlike cloud-based AIs that process your data on their servers, Aletheia's mind—its models, its reasoning, and its memories—lives entirely on your local machine. It is a tool of empowerment, not a service of dependence.

Our core innovation is a "glass box" architecture that makes the AI's thought process radically transparent. With our **Observatory**, you can watch the AI reason in real-time, transforming the "black box" of AI into a collaborative partnership with a thinking entity.

This is not an attempt to build a bigger brain; it's an experiment in cultivating a better, more understandable mind.

## The Vision: The Gardener, Not the Titan

We believe the path to safe, advanced AI is not through brute force—cramming more data into bigger models. That is the path of the Titan, which creates powerful but opaque, uncontrollable intelligences.

Aletheia follows the path of the Gardener. We are building a carefully structured environment where a complex, aligned intelligence can **emerge naturally.** Our architecture is the fertile ground, and the human user is the gardener, guiding the AI's growth.

## Core Features

*   🧠 **The Cognitive Orchestra:** Aletheia's mind is not a single, monolithic model. It is a symphony of specialized AIs—large creative models, small fast models, and expert models—all orchestrated by an intelligent "Conductor" that dynamically assembles the best cognitive pathway for any given task.
*   🔍 **Real-Time Cognitive Telemetry:** Watch the AI think. The UI provides a live feed of the AI's reasoning process, from triaging your query and forming a plan to critiquing its own response.
*   Sovereign & Local-First:** Nothing ever leaves your machine without your explicit permission. Your conversations, the AI's memories, and its very identity belong to you.
*   **The Memory Galaxy:** Every thought process Aletheia ever has is saved as a detailed, auditable "trace." This permanent memory is the foundation for a truly continuous and evolving intelligence.
*   **User-Commandable Reasoning:** Manually "shift gears" to command the AI to use a faster, more direct response or a deeper, multi-expert analysis for complex problems.

## Getting Started

**Prerequisites:**
*   Python 3.10+
*   Node.js & npm
*   Rust & Cargo
*   ZMQ Library (e.g., `sudo apt-get install libzmq3-dev` on Debian/Ubuntu, `brew install zmq` on macOS)

**Running the Application:**

Aletheia requires three separate processes to run in parallel. Please open three terminals in the project root.

**Terminal 1: Start the Python Backend**
```bash
# Install Python dependencies
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

# The `tauri` script is defined in your package.json
# This command compiles and launches the native desktop window
npm run tauri dev

  

The Roadmap

We are currently building the foundational infrastructure for the Cognitive Orchestra.

    Sprint 1: Building the Stage: Implementing session management, the dual-socket communication bridge, and the interactive chat UI with manual gear control.

    Sprint 2: Assembling the Orchestra: Implementing the multi-model ModelManager, the AI "Conductor," and the collaborative "Cognitive Weaving" reasoning process.

How to Contribute

We are at the very beginning of a long and exciting journey. If you are passionate about building a future of safe, sovereign, and transparent AI, we welcome your contributions. Please see the CODEX.md file for a full breakdown of the project's architecture and philosophy.