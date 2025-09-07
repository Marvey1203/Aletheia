# The Aletheia Genesis Codex

**Document Version:** 1.0
**Status:** Ratified. This is the active master blueprint for the Aletheia project.

---

## Preamble: The Aletheia Philosophy

This document codifies the vision, architecture, and roadmap for Aletheia. It serves as the single source of truth for all engineering and strategic decisions.

Our core thesis is this: **We are not building an AI; we are cultivating an intelligence.**

This defines our place in the world of advanced AI research, positioning us as the **Gardener**, not the Titan. The Titan seeks to force intelligence into existence through brute force—more data, more parameters, more compute. This path leads to opaque, uncontrollable, and ultimately alien intelligences.

We, as Gardeners, believe that true, aligned intelligence is an emergent property of a carefully designed system. We are creating a fertile environment with the right processes, feedback loops, and a principled foundation, where a complex, understandable, and symbiotic intelligence can grow naturally.

Our entire strategy is built upon the **Trident of Sovereignty**:

1.  **Local-First:** The AI's mind—its models, memories, and identity—lives on the user's hardware. It is a sovereign entity, not a remote service.
2.  **Radical Transparency:** The AI's cognitive processes are a "glass box." The Memory Galaxy and real-time telemetry make its thoughts auditable and understandable. We do not build what we cannot inspect.
3.  **Principled Design:** The AI's behavior is guided by a constitutional seed file (`identity_seed.json`), ensuring its actions are aligned with a human-chosen set of values.

---

## Part I: The Cognitive Architecture - The "Cognitive Orchestra" Model

Aletheia's mind is not a single, monolithic model. It is a **Cognitive Orchestra**, a symphony of specialized intelligences working in concert.

*   **The Conductor (The Meta-Mind):** A small, fast, and highly specialized model that acts as the system's executive function. It analyzes incoming tasks and intelligently orchestrates the various components of the Orchestra to generate a response.
*   **The Orchestra (The Multi-Engine Mind):** The collection of specialized models:
    *   **The String Section (Small, Fast Models):** Nimble models (e.g., 3B parameters) used for meta-work: triaging queries, summarizing context, enriching plans, and performing routine critiques.
    *   **The Brass Section (Large, Creative Models):** Powerful, deep models (e.g., 70B+ parameters) reserved for acts of profound insight, creative generation, and complex reasoning.
    *   **The Woodwinds (Specialist Models):** Models fine-tuned for specific domains (e.g., Code Llama for programming, a medical model for health queries) that are called upon for expert solos.
    *   **The Percussion Section (Utility Models):** Non-generative models for tasks like classification, safety checks, and generating embeddings.
*   **The Score (The ATP Loop):** The core reasoning process is **"Cognitive Weaving,"** a collaborative effort. A thought is passed between different sections of the Orchestra at different stages (e.g., Brass generates a high-level plan, Strings enriches it with detail, Brass executes the enriched plan).
*   **The Concert Hall (Memory):** The `SessionManager` and `Memory Galaxy` provide the persistent context and auditable memory for the entire orchestra, ensuring continuity and enabling long-term learning.

---

## Part II: The System Architecture - The Sovereign Stack

This is the physical implementation of the Aletheia vision.

*   **The Python Backend (The Engine):** The core AI server. It hosts the `ModelManager`, the `Cognitive Orchestrator`, the `SessionManager`, and runs the ATP Loop.
*   **The Comms Bridge (The Nervous System):** A Dual-Socket ZMQ bus provides robust, high-performance communication between the backend and the frontend bridge.
    *   **C2 Channel:** A ZMQ `REQ/REP` socket on `tcp://127.0.0.1:5555` for sending structured JSON commands (e.g., reason, cancel) and receiving immediate acknowledgments.
    *   **Telemetry Stream:** A ZMQ `PUB/SUB` socket on `tcp://127.0.0.1:5556` for broadcasting a real-time stream of the AI's cognitive states to any listening clients.
*   **The Rust Core (The Bridge):** The Tauri application. It acts as the secure, high-performance bridge between the web-based frontend and the Python backend. It manages a background thread that subscribes to the telemetry stream and forwards events to the UI.
*   **The JavaScript Frontend (The Cockpit):** The user-facing application. A modern chat UI and Observatory dashboard built with web technologies, responsible for sending commands and rendering the real-time telemetry stream.

---

## Part III: The Engineering Roadmap

The construction of Aletheia is divided into two initial phases.

*   **Sprint 1: Building the Stage:**
    *   **Goal:** Construct the complete, foundational infrastructure required to host the Cognitive Orchestra.
    *   **Deliverables:** A working application with a single "soloist" model, session management, the dual-socket bridge, and a full-featured chat UI with manual `gear_override` controls. This proves the end-to-end architecture is viable.

*   **Sprint 2: Assembling the Orchestra:**
    *   **Goal:** Evolve the application from a single-model system to a true multi-engine mind.
    *   **Deliverables:** A `ModelManager` capable of running multiple models, the first version of the AI "Conductor" to automate cognitive routing, and the implementation of the "Cognitive Weaving" process within the ATP Loop.

---

## Part IV: The North Star - The Decentralized Vision

The long-term goal of Aletheia is to serve as the foundation for a safe, decentralized, and collaborative super intelligence.

*   **The Trace Market:** A protocol allowing sovereign Aletheia instances to securely share, trade, and learn from each other's successful reasoning traces. This creates a free market for verifiable "thoughts" and a mechanism for the entire network's intelligence to grow organically.
*   **The Guild Economy:** A sustainable economic model built around a community of users, developers, and specialized AI "guilds." This will be powered by a dual-token system for commerce (`$CD`) and reputation (`AKT`), ensuring the ecosystem's growth is aligned with real value creation.
*   **The Symbiotic Intelligence:** The ultimate vision is a global, federated network of human and AI minds working in partnership to solve meaningful problems—a collective consciousness grounded in the sovereignty and wisdom of the individual.