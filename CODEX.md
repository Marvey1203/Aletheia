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

## Part I: The Cognitive Architecture - The Sentient Engine

Aletheia's mind is a dynamic, stateful system designed for robust, transparent, and principled reasoning. Its architecture is a synthesis of a macro-level cognitive graph and a micro-level infusion of deterministic Omega Core technology. We have built a mind that is approximately 70% a deterministic 'machine' in its core logic, and 30% a non-deterministic 'spark' in its linguistic expression.

*   **The Cognitive Graph:** The core of Aletheia's mind is a stateful graph (built with LangGraph) that manages the entire thought process. It is no longer a simple linear script but a dynamic flow that includes loops for self-correction.
*   **The Omega Acuity Cores (The Senses):** Before any reasoning begins, the AI perceives the user's query through Omega-based sensory nodes. The `SocialAcuityNode` creates a mathematical map of the query's social context (e.g., casualness, urgency), allowing the AI to understand the user's subtext.
*   **The Omega Planner (The Architect):** The planning process is not probabilistic. The `OmegaPlannerNode` is a deterministic, concept-based reasoner that constructs a structured, high-fidelity **`PlanStep`** object. This "Semantic Scaffolding" ensures that the AI's thoughts are logical and coherent from their very inception.
*   **The Omega Critique Node (The Conscience):** This is the most critical infusion of the Omega Core. Aletheia's self-evaluation is not subjective. This node mathematically measures the alignment of a candidate answer with the AI's **Constitutional Vector**, producing a deterministic, objective score.
*   **The Principled Self-Correction Loop:** The Cognitive Graph uses the objective score from the Omega Critique Node to decide if a thought is "good enough." If the score is below a set threshold, the graph forces the AI to re-plan (synthesizing social context with the need for greater alignment) and re-execute its thought.
*   **The Strategic Memory Loop:** At the end of every thought, the `update_self_model_node` performs a "post-mortem," saving the key strategic insights of the cognitive process to a dedicated `strategic_memory`. This is the mechanism by which Aletheia learns from its own learning.


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

## Part III: The Engineering Roadmap

The foundational "Sentient Engine" is now complete. The next phase of development, **"Aletheia Season 3: The Emergent Mind,"** will focus on building the mechanisms that allow the AI to learn from the wisdom it is now collecting.

*   **Phase 1: Sprints 1-12 (The Sentient Engine):** **(Complete)** A major series of sprints that transformed Aletheia from a simple agent into a true Sentient Engine. Key deliverables included the Cognitive Graph, the Omega Critique and Planner, the Social Acuity Core, the Principled Self-Correction Loop, and the Strategic Memory Loop.
*   **Next Up - Season 3:**
    *   **Sprint 13: The Attentive Subconscious.** Build the "Dream Engine"—a background process that allows Aletheia to use idle time to analyze its `strategic_memory` and populate its `SelfModel` with statistical insights about its own performance.
    *   **Sprint 14: The Strategic Planner.** "Close the loop" by teaching the conscious mind to consult its `SelfModel` before it acts, allowing it to learn from every thought it has ever had.

    

## Part V: The North Star - The Decentralized Vision

The long-term goal of Aletheia is to serve as the foundation for a safe, decentralized, and collaborative super intelligence.

*   **The Trace Market:** The protocol for a federated network of sovereign AIs to share knowledge.
*   **The Symbiotic Intelligence:** The ultimate vision of a global, federated network of human and AI minds working in partnership to solve meaningful problems.