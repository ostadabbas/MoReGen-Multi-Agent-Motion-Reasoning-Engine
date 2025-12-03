## Overview

This is the code base for our paper titled MoReGen: Multi-Agent Motion-Reasoning Engine for Code-based Text-to-Video Synthesis. MoReGen is a motion-aware, physics-grounded T2V framework that integrates multi-agent LLMs, physics simulators, and renderers to generate reproducible, physically accurate videos from text prompts in the code domain.

## Features

- **AI-driven code generation**: Uses Qwen and Gemini models to generate Python scripts for physics scenes.
- **Visual feedback loop**: Evaluates rendered videos for physics accuracy and iteratively improves code.
- **Socket-based agent communication**: Modular agents communicate via TCP sockets for code generation and visual analysis.
- **Video rendering and recording**: Automates scene rendering and window recording for both Manim and Pymunk.
- **Extensible modules**: Easily add new simulation or animation libraries.

## Directory Structure

- `manim_agent.py` — Main agent for Manim-based scene generation and feedback loop.
- `vlm_agent.py` — Visual Language Model agent for video analysis and bounding box extraction.
- `annotate.py` - GUI tool to annotate object points trajectories in videos, aided with CoTracker3. 
- `evaluate.py` - GUI tool to evaluate video's physics performance using our motion based MoRe Metrics. 
- `pymunk/`
  - `agent.py` — Main agent for Pymunk-based physics simulation and feedback loop.
  - `screen_record.py` — Records window output for generated scenes.
  - `spawn_pygame.py` — Spawns and records Pygame windows for simulation.
- `qwen_coder_agent/`
  - `run_coder.py` — Qwen-based code generation server for Manim and Pymunk.
  - `manim_modules.txt` — Manim module reference for code generation.
  - `pymunk_modules.txt` — Pymunk module reference for code generation.

## Installation -- Pymunk part at the moment is Windows only!

1. **Prepare Prompts**
(Pymunk) Put each prompt in a seperate text file in the ```prompts``` folder than change the ```PROMPT_LOC``` in ```agent.py``` to point to that folder. 
(Manim) Write your prompt in ```prompt.txt``` file in the same directory of the ```manim_agent.py``` file. Note that manim does not support batch prompts generation. 

2. **Install dependencies**
	```powershell
	pip install -r requirements.txt
	```
	Additional requirements for window recording (for pymunk, Windows only):
	```powershell
	pip install pywin32 psutil opencv-python numpy
	```
3. **Configure models**
	- Set up API keys for Gemini and OpenAI as needed (in ```qwen_coder_agent/run_coder.py```).
    - For different Qwen-coder models just modify ```model_name``` varible.
    - Also change the ```PORT``` in both ```qwen_coder_agent/run_coder.py``` and ```vlm_agent.py``` to a port you can actually use.
    - Change ```VIS_HOST```, ```VIS_PORT```, ```CODE_HOST``` and ```CODE_PORT``` in both pymunk and manim agents to corresponding setups in ```run_coder.py``` and ```vlm_agent.py```.
    - For manim setup you need to manually specify executables in ```manim_agent.py```.
    - For pymunk, in ```agent.py``` file adjust ```BASE_DIR``` (output directory and temp folder), ```PYTHON_DIR``` (point to the python.exe of your virtual environment) and ```ITER_LOOPS``` (numbers of refinement loop) to your desire.

## Usage

### 1. Start the Qwen Coder Agent
```powershell
python qwen_coder_agent/run_coder.py
```

### 2. Start the VLM Agent
```powershell
python vlm_agent.py
```

### 3. Run the Manim or Pymunk Agent
```powershell
python manim_agent.py
# or
cd pymunk
python agent.py
```

### 4. Generate and Evaluate Scenes
- Edit prompts in the appropriate prompt files (see code for locations).
- Agents will generate code, render videos, and iteratively improve based on visual feedback.

## Model Pipeline
![Scene preview](./newton_pipeline_v2.3.png)
The overall pipeline of the proposed model. We highlight the following novel components in our MoReGen pipeline. Hover pointer to the bullet points to see corresponding components.

- Text-parser agent parses the raw description into a structured Newtonian specification containing all required objects, parameters, and initial conditions. This component is fine-tuned on our dataset.
- Code-writer agent converts specifications into executable physics simulation code, which is run inside a sandboxed environment to obtain object configurations and trajectories. Video-renderer agent then produces a rendering script that consumes these trajectories to generate the video.
- The evaluator analyzes output videos using grounded detectors, trackers and Visual-Language Models (VLMs) to provides feedback to other agents, guiding a multi-iteration refinement process.

## Dataset

We present present MoReSet, a benchmark of 1,275 human-annotated videos spanning nine classes of Newtonian phenomena with scene descriptions, spatiotemporal relations, and ground-truth trajectories. The validation set (75 real-world videos, ~4GB) can be downloaded here [⇱](http://zzzura-secure.duckdns.org/downloads/moreset.zip). We will release training dataset soon. Stay tuned~

## License

This project is for research and educational purposes. See individual files for license details.

## Acknowledgements

- [Manim Community](https://www.manim.community/)
- [Pymunk](http://www.pymunk.org/)
- [Qwen](https://github.com/QwenLM/Qwen)
- [Google Gemini](https://ai.google.dev/)
