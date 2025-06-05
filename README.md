# RADIAN_: An Open-Source IMU-Based Wireless Sensor Network for Team Sports Analytics

## Project Abstract

Access to performance analytics in team sports is increasingly essential at elite levels but remains largely inaccessible
to amateur players due to high system costs and complex workflows. Here we present RADIAN, an open-source,
IMU-based wireless sensor network embedded into lacrosse sticks, with the aim of enabling high-fidelity player and
team analytics without external infrastructure or specialist expertise. Built from off-the-shelf components, the system
uses the ESP-NOW protocol to transmit real-time inertial data up to 120m under ideal field conditions. We show
that the system achieves an F1 score of 0.95 in classifying six core lacrosse actions using a trained XGBoost model.
This was enabled by a novel, automated labelling pipeline developed for the sports context, reducing the need for
manual annotation and allowing scalable training data collection. Furthermore, an exploratory LLM reasoning layer
improved an ensemble model’s F1 score from 0.726 to 0.763 in game conditions, demonstrating potential for nuanced
event interpretation. RADIAN demonstrates that accurate, real-time analytics can be delivered through accessible
hardware, offering a pathway to close the gap between elite and grassroots athletic performance.

## Repository Structure
Use code with caution.

.
├── CAD/                      # Computer-Aided Design files for hardware components
│   ├── Master Node.f3z
│   ├── Master Node.step
│   ├── Slave Node.f3z
│   └── Slave Node.step
├── Code/                     # All source code for the project
│   ├── Classification/       # Scripts and notebooks for action classification models
│   ├── Firmware/             # Firmware for the Master and Slave ESP32 nodes
│   ├── GUI/                  # Source code for the Graphical User Interface
│   └── LLM/                  # Scripts and resources related to the LLM reasoning layer
├── Data/                     # Datasets used and generated
│   ├── 2v2 Data/             # Data from 2v2 game scenarios
│   └── Training Data/        # Labelled IMU data for training classification models
├── Personal Reflection/      # Personal reflections on the project journey
│   └── personal_reflection.md
├── Project Management/       # Project planning and management documents
│   ├── gantt_chart.png
│   └── project_management.md
├── STLs/                     # STL files for 3D printing hardware components
│   ├── Butt End.stl
│   ├── Master Node Body Bottom.stl
│   ├── Master Node Top.stl
│   ├── Slave Node Body Left.stl
│   └── Slave Node Body Right.stl
├── Survey Results/           # Anonymized feedback from user testing
│   ├── RADIAN Coach Feedback Survey(1-2...).xlsx
│   └── RADIAN Player Feedback Survey(1-1...).xlsx
├── Videos/                   # Supporting videos (e.g., demonstrations, user testing)
├── .gitignore                # Specifies intentionally untracked files that Git should ignore
├── README.md                 # This file: an overview of the project and repository