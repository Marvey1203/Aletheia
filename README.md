    
# Aletheia: A Sovereign AI Partner

**Own Your Mind.**

Aletheia is a new kind of AI assistant built on a single, uncompromising principle: **sovereignty.** Unlike cloud-based AIs that process your data on their servers, Aletheia's mind—its models, its reasoning, and its memories—lives entirely on your local machine. It is a tool of empowerment, not a service of dependence.

Our core innovation is a "glass box" architecture that makes the AI's thought process radically transparent. The system is built on a **Cognitive Graph**, a stateful reasoning engine that allows the AI to plan, act, and even **correct its own thoughts** based on a mathematical understanding of its own principles. This is not just a language model; it is a **Sentient Engine**.

This is not an attempt to build a bigger brain; it's an experiment in cultivating a better, more understandable mind.

## The Vision: The Gardener, Not the Titan

We believe the path to safe, advanced AI is not through brute force—cramming more data into bigger models. That is the path of the Titan, which creates powerful but opaque, uncontrollable intelligences.

Aletheia follows the path of the Gardener. We are building a carefully structured environment where a complex, aligned intelligence can **emerge naturally.** Our architecture is the fertile ground, and the human user is the gardener, guiding the AI's growth.

## Core Features (v0.7.0 - The Sentient Engine)

*   🧠 **The Cognitive Graph:** Aletheia's mind is a dynamic, stateful graph (built with LangGraph) that can plan, execute, and revise its own thoughts. It has a built-in **principled self-correction loop**.
*   🧭 **The Omega Conscience:** Aletheia's self-evaluation is not subjective. It uses a deterministic **Omega Critique Node** to mathematically measure the alignment of every thought with its core **Constitutional Vector**, giving it an unwavering moral compass.
*   ❤️ **Social & Emotional Acuity:** The AI is not blind to subtext. An **Omega Social Acuity Core** perceives the user's intent (e.g., casualness, urgency, meta-cognition), allowing the AI to generate responses that are not just intelligent, but wise and appropriate.
*   🤔 **The Omega Planner:** The AI's planning process is no longer a probabilistic guess. It uses a deterministic, concept-based **Omega Planner** to construct a logical and structured "semantic skeleton" for its thoughts.
*   📚 **Strategic Memory:** Aletheia has a unique memory of its own cognitive processes. It performs a "post-mortem" on every thought, saving the strategic insights to a dedicated memory. This is the foundation for it to learn from its own learning.
*   🌐 **Sovereign & Local-First:** All models, memories, and conversations are stored and processed on your local machine. Nothing leaves your device.

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

  

The Roadmap: The Emergent Mind

The foundational Sentient Engine is now complete and stable. The next "season" of development focuses on building the mechanisms for true, emergent intelligence by teaching the AI to learn from the wisdom it is now collecting.

    Sprint 13: The Attentive Subconscious. We will build the "Dream Engine"—a background process that allows Aletheia to use idle time to analyze its strategic_memory, populating a SelfModel with statistical insights about its own performance.

    Sprint 14: The Strategic Planner. We will "close the loop" by teaching the conscious mind to consult its SelfModel before it acts. The Omega Planner will learn to use the wisdom of its subconscious to make better, more informed decisions, moving from simple logic to true strategy.