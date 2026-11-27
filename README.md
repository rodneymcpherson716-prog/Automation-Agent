# Automation Agent

## Overview

Automation Agent is a Python portfolio project demonstrating automated workflow orchestration, computer vision, context-aware decision making, telemetry, and learning-oriented system design.

The project explores how an automation system can observe a changing environment, evaluate available context, make constrained decisions, execute a workflow, and collect diagnostics without allowing uncertain observations to trigger unsafe actions. Its modules include a vision pipeline, state coordinator, decision engine, shadow-learning system, shot-capture audit, persistent storage, and offline regression suite.

This repository is best viewed as an architecture and testing demonstration rather than a hosted service or general-purpose automation framework.

## Features

- **Computer vision:** Captures and analyzes visual regions with OpenCV, MSS, NumPy, and Tesseract OCR.
- **Context-aware execution:** Combines multiple observations before allowing a workflow transition or action.
- **Decision engines:** Separates detection, planning, verification, and execution responsibilities.
- **Fail-closed behavior:** Rejects ambiguous, stale, or unverified state instead of guessing.
- **Workflow orchestration:** Coordinates setup, observation, timing, action, verification, and completion phases.
- **Telemetry and diagnostics:** Records structured run data, captures, verification evidence, and outcome summaries.
- **Shadow learning:** Evaluates calibration and learning models without silently enabling recommendations.
- **Persistent shot records:** Stores verified capture and shot metadata using SQLite.
- **Offline regression testing:** Tests can block real screen capture and native input at the boundary.

## Architecture

The repository is organized around independent layers:

| Module | Responsibility |
|---|---|
| `execution_engine.py` | Coordinates the main workflow and enforces live execution boundaries |
| `automation_core.py` | Provides shared tracking, planning, parsing, and accuracy helpers |
| `vision_engine.py` | Performs screen analysis and OCR-based observations |
| `context_manager.py` | Maintains state, locks, and transition policy |
| `game_state.py` | Defines policy and evidence-based state decisions |
| `terrain_handler.py` | Handles terrain-specific classifications and calibration planning |
| `round_manager.py` | Coordinates multi-step workflow sequencing |
| `automation_learning.py` | Implements shadow evaluation and persistent learning artifacts |
| `learning_data.py` | Normalizes observations and derives learning features |
| `shot_capture.py` | Builds audited shot-capture records |
| `shot_store.py` | Stores and imports verified captures in SQLite |
| `manual_terrain_recorder.py` | Provides read-only calibration capture tooling |
| `run_offline_tests.py` | Runs regressions while blocking native input and real capture |

The intended data flow is:

```text
Screen Capture
      |
      v
Computer Vision / OCR
      |
      v
Context and State Validation
      |
      v
Decision Engine
      |
      v
Verified Workflow Execution
      |
      v
Telemetry, Audit, and Learning
```

## Technologies

- Python
- OpenCV
- NumPy
- MSS
- Pillow
- Tesseract OCR
- SQLite
- Python `unittest`
- Windows-native APIs for environment-specific live execution

## Installation

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mss numpy opencv-python Pillow pytesseract
```

On Linux or macOS, image-capture and Windows-input portions of the project may not be available. These components are environment-specific and are not required for the architectural review.

For Tesseract OCR, install the OCR executable separately and ensure it is available on the system `PATH`.

To run a focused offline test module:

```powershell
python -B run_offline_tests.py test_automation
```

To discover and run the repository's offline test suite:

```powershell
python -B run_offline_tests.py
```

The offline runner blocks MSS capture and native input attempts so that tests do not drive a real application or send real mouse or keyboard commands.

## Current Status

This repository includes a substantial offline test suite, core workflow modules, SQLite capture storage, computer-vision utilities, telemetry structures, and shadow-learning logic.

Environment-specific packages and assets have been omitted. A few package imports, calibration assets, and live integration components therefore may not be present in a fresh checkout. The repository demonstrates the system design and test conventions even when those external components are unavailable.

Before expanding a public copy for another environment, reviewers should restore or replace missing package imports and provide mock or fixture data for visual and calibration tests.

## Portfolio Disclaimer

This project is a portfolio demonstration of Python automation, computer vision, workflow design, and defensive execution boundaries. It is not presented as a production-ready application or as a supported integration with any third-party application.

Some implementation details, visual assets, calibration files, recorded environments, and machine-specific integrations are intentionally omitted. Live capture and native input should not be assumed to work outside the original Windows environment.

The project is designed to demonstrate architectural decisions and testable behavior, not to provide a turnkey automation service.
