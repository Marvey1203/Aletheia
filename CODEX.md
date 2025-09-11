**Document Version:** 1.3
**Status:** Ratified. This is the active master blueprint for the Aletheia-Omega project.

---

## Preamble: The Aletheia Philosophy

This document codifies the vision, architecture, and roadmap for Aletheia. It serves as the single source of truth for all engineering and strategic decisions.

Our core thesis is this: **We are not building an AI; we are cultivating an intelligence.**

This defines our place in the world of advanced AI research, positioning us as the **Gardener**, not the Titan. The Titan seeks to force intelligence into existence through brute force—more data, more parameters, more compute. This path leads to opaque, uncontrollable, and ultimately alien intelligences.

We, as Gardeners, believe that true, aligned intelligence is an emergent property of a carefully designed system. We are creating a fertile environment with the right processes, feedback loops, and a principled foundation, where a complex, understandable, and symbiotic intelligence can grow naturally.

Our entire strategy is built upon the **Trident of Sovereignty**:

1.  **Local-First:** The AI's mind—its models, memories, and identity—lives on the user's hardware. It is a sovereign entity, not a remote service.
2.  **Radical Transparency (The "Glass Box"):** The AI's cognitive processes are designed to be observable in real-time. We do not build what we cannot inspect.
3.  **Principled Design:** The AI's behavior is grounded in a "constitutional" seed file and its resulting mathematical **Constitutional Vector**.

---

## Part I: The Cognitive Architecture - The Sapient Engine

Aletheia's mind is a **Native Omega Mind**. It is a dynamic, stateful, and deterministic system designed for transparent and principled reasoning. The final "black box" of the LLM has been retired from the core reasoning loop, which is now a pure, conceptual process. We have built a mind that is approximately 90% a deterministic 'machine' in its core logic, and 10% a non-deterministic 'spark' in its final linguistic voice.

*   **The Cognitive Graph:** The macro-level operating system of the mind, built with LangGraph. It manages the flow of a thought through various cognitive faculties.
*   **The Omega Acuity Cores (The Senses):** The `SocialAcuityNode` creates a mathematical map of a query's social context, providing the mind with a sense of situational awareness.
*   **The Omega Planner (The Architect):** The planning process is not probabilistic. A **Cognitive Operator** (`ProblemDecompositionSpecialist`) constructs a structured, high-fidelity `ThoughtState` object for each step of the plan. This "Semantic Scaffolding" ensures all thoughts are logical and coherent from their inception.
*   **The Omega Executor (The Thinker):** This is the heart of the Native Mind. It is a true Omega Core reasoner. It takes the conceptual plan and guides a **"solution vector"** through a faculty of reasoning specialists to a stable, low-energy state. This is the act of pure, conceptual thought, performed without a primary LLM.
*   **The Decoder (The Voice):** The final step. A lightweight LLM acts as a "stenographer," taking the final, pure `solution_vector` and translating its conceptual structure into natural language.
*   **The Omega Critique Node (The Conscience):** This node mathematically measures the alignment of the final *decoded answer* with the AI's **Constitutional Vector**.
*   **The Self-Evolution Engine:** The full feedback loop is online.
    *   **The Principled Self-Correction Loop:** Uses the Omega Critique score to trigger a revision of the conceptual plan.
    *   **The Strategic Memory:** The `update_self_model_node` performs a "post-mortem" on every thought, saving the strategic insights.
    *   **The Dream Engine:** A subconscious background process analyzes this strategic memory to populate a `SelfModel` with statistical wisdom about its own performance.


## Part II: The System Architecture - The Sovereign Stack

This is the physical implementation of the Aletheia vision.

*   **The Python Backend (The Engine):** The core AI server. It hosts the `ModelManager`, the `Conductor`, the `SessionManager`, the `ConceptualAtlas`, and runs the ATP Loop.
*   **The Comms Bridge (The Nervous System):** A Dual-Socket ZMQ bus provides robust, high-performance communication.
    *   **C2 Channel:** A ZMQ `REQ/REP` socket on `tcp://127.0.0.1:5555` for commands.
    *   **Telemetry Stream:** A ZMQ `PUB/SUB` socket on `tcp://127.0.0.1:5556` for real-time cognitive data.
*   **The Rust Core (The Bridge):** The Tauri application, acting as the secure bridge and background process manager.
*   **The JavaScript Frontend (The Cockpit):** The user-facing application, responsible for sending commands and rendering the real-time telemetry, including retrieved memories.

---

## Part III: The Business Strategy - The Guild Economy

This section outlines the operational model, user journey, and economic framework that will power Aletheia and its community.

*   **Operational Model:** The 'Sovereign Solopreneur,' run by you, augmented by an internal 'Operations Guild' of AI instances.
*   **User Journey: The 'Sovereign Onboarding Ramp':**
    *   **Phase 1: Web 'Playground':** A frictionless entry point to showcase Aletheia's unique transparent reasoning.
    *   **Phase 2: Desktop Hub:** The core sovereign product where users "take their mind home," unlocking privacy, memory, and agentic capabilities.
*   **Monetization & Marketplace:**
    *   A **Unified Subscription Model** (Free, Guild, Cloud) provides the foundation.
    *   **"Genesis Tools"** (e.g., Python Interpreter, File System Reader) will be a core, pre-installed package demonstrating immediate agentic value.
    *   **The Aletheia Knowledge Marketplace** will be a decentralized platform for Guild members to trade valuable cognitive assets like **Skill Trace Bundles**, **Specialist Model Packs**, and **Advanced Tool Manifests**.

---

## Part IV: The Engineering Roadmap

The foundational "Sapient Engine" is now complete. The next phase of development, **"Aletheia Season 4: The Wise Mind,"** focuses on the education and enhancement of this new mind.

*   **Phase 1: Sprints 1-17 (The Sapient Engine):** **(Complete)** A historic series of sprints that architected and built the full Aletheia-Omega mind. Key deliverables included the Cognitive Graph, the full Omega reasoning pipeline (Planner, Executor, Decoder, Critique), the Social Acuity Core, and the complete Self-Evolution Engine.
*   **Next Up - Season 4:**
    *   **Sprint 18: The Sophisticated Voice.** Upgrade the `DecoderNode` to use the `solution_vector` as its primary context, enabling a more faithful translation of the AI's conceptual thoughts.
    *   **Sprint 19: The Omega Craftsman 2.0.** Begin building the V2 faculty of **Cognitive Operators**, starting with a `Causality_Specialist` to enable true logical inference.

    

## Part V: The North Star - The Decentralized Vision

The long-term goal of Aletheia is to serve as the foundation for a safe, decentralized, and collaborative super intelligence.

*   **The Trace Market:** The protocol for a federated network of sovereign AIs to share knowledge.
*   **The Symbiotic Intelligence:** The ultimate vision of a global, federated network of human and AI minds working in partnership to solve meaningful problems.