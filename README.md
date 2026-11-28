# Automation Agent

Automation Agent is a Python-based automation project focused on computer vision, workflow automation, state detection, and reliable interaction with a graphical desktop application.

The project demonstrates how an automation system can observe changing screen conditions, interpret application state, make decisions, and execute controlled actions while maintaining safety checks and testable behavior.

## Key Features

- Python-based automation architecture
- Computer vision and screen-state detection
- Multi-step workflow automation
- Application and window recognition
- Context-aware decision logic
- Multi-monitor support
- Automated input and action sequencing
- State and transition handling
- Debug and diagnostic logging
- Offline regression testing
- Emergency-stop and safety controls
- Structured configuration and calibration data

## Engineering Approach

The project was developed as a modular automation system rather than a single script.

Different components handle responsibilities such as screen observation, application-state detection, decision making, action execution, calibration, learning data, and testing. This separation makes the project easier to troubleshoot, test, maintain, and improve.

The automation also accounts for changing application conditions instead of relying entirely on fixed coordinates or static assumptions. This allows the system to make decisions based on information detected from the current screen state.

## Reliability and Testing

Reliability was an important part of development.

The project includes regression tests and offline validation to help identify problems before live execution. Recent migration validation included Python compilation checks, structured-data validation, database integrity checking, and focused automated tests.

Safety controls are also built into the workflow to prevent unintended actions and provide a way to stop automation when necessary.

## Project Migration

The repository was recently reorganized under the **Automation Agent** name to provide a cleaner, more professional project structure.

The migration included:

- Source-file and documentation renaming
- Reference and dependency updates
- Configuration and data-file migration
- Test updates
- Repository cleanup
- Validation of Python and structured-data files
- Regression testing after migration

Application-specific runtime identifiers required for correct detection and compatibility are intentionally preserved inside the implementation where necessary.

## Technologies

- Python
- Computer Vision
- Desktop Automation
- Workflow Automation
- Git and GitHub
- JSON / structured configuration
- SQLite
- Automated Testing

## What This Project Demonstrates

This project highlights practical experience with:

- Automation engineering
- Python development
- Computer vision concepts
- Debugging complex workflows
- State-machine and decision logic
- Test-driven troubleshooting
- Git version control
- Repository migration and refactoring
- Maintaining compatibility during large codebase changes

## Author

**James McPherson**

Cybersecurity and technology professional with an interest in automation, systems engineering, infrastructure, and security-focused software development.
