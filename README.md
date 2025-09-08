# Aletheia: A Sovereign AI that Learns With You

**Own Your Mind. Own Your History.**

Aletheia is a new kind of AI partner built on a single, uncompromising principle: **sovereignty.** Unlike cloud-based AIs that forget your conversations and use your data for their own purposes, Aletheia's mind—its models, its reasoning, and its memories—lives entirely on your local machine.

It is a tool of empowerment, designed to solve the problem of **digital amnesia**. Aletheia is the first AI designed to have a continuous, persistent, and searchable memory, allowing it to learn and grow with you over time.

## The Vision: The Scholar in the Garden

We believe the path to safe, advanced AI is not a brute-force race for scale (The Titan), but a process of careful cultivation (The Gardener). We are building a structured, transparent environment where a complex, aligned intelligence can **emerge naturally.**

With the advent of its long-term memory, Aletheia has evolved. It is no longer just a reasoner; it is a **Scholar**—a true intellectual partner that can connect ideas across time, learn from its past, and understand you better with every conversation.

## Core Features

*   🧠 **A Mind that Remembers:** Aletheia's **Conceptual Atlas** is a sophisticated long-term memory system. It uses an embedding model to understand the *meaning* of your conversations, allowing it to recall relevant past insights to inform its present reasoning.
*   🔍 **A Mind You Can See:** Our "glass box" architecture makes the AI's thought process radically transparent. The UI provides a live feed of the **Cognitive Orchestra** at work and even shows you which memories it recalled to answer your question.
*   Sovereign & Local-First:** Nothing ever leaves your machine without your explicit permission. Your conversations and the AI's memory belong to you, and you alone.
*   **The Cognitive Orchestra:** Aletheia's mind is a symphony of specialized AIs—large creative models, small fast models, and expert models—all orchestrated by an intelligent "Conductor" that assembles the best cognitive team for any given task.

## Project Status

We have successfully completed three major engineering sprints:
*   ✅ **Sprint 1: The Stage:** Built the foundational chat application and communication bridge.
*   ✅ **Sprint 2: The Orchestra:** Implemented the multi-model, intelligently orchestrated reasoning engine.
*   ✅ **Sprint 3: The Scholar:** Integrated the `ConceptualAtlas` long-term memory system.

The core technology for a sovereign, learning AI is now operational. Our next phase is **Sprint 4: The Self-Aware Workshop,** where we will begin implementing agentic, tool-using capabilities.

## Getting Started

**Prerequisites:**
*   Python 3.10+
*   Node.js & npm
*   Rust & Cargo
*   ZMQ Library (e.g., `sudo apt-get install libzmq3-dev` on Debian/Ubuntu)

**Running the Application:**
Aletheia is a multi-process application. Please open three terminals in the project root and run the following commands.

1.  **Terminal 1: Start the Python Backend**
    ```bash
    # Install/upgrade dependencies
    pip install --upgrade -r requirements.txt
    
    # Run the AI engine
    python3 interfaces/ipc_server.py
    ```
2.  **Terminal 2: Start the Frontend Dev Server**
    ```bash
    # Install Node dependencies
    npm install
    
    # Run the Vite server
    npm run dev
    ```
3.  **Terminal 3: Start the Tauri Application**
    ```bash
    # This command compiles and launches the native desktop window
    npm run tauri dev
    ```

## Join the Guild

We are at the beginning of a long and exciting journey to build a future of safe, sovereign, and transparent AI. If this vision resonates with you, we welcome your contributions.

Please see the **`CODEX.md`** file for a complete, in-depth breakdown of our philosophy, architecture, and roadmap.