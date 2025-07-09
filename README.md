# Extended Currentscapes (in vitro demo)

This repository demonstrates an extension of the *currentscape* method for analyzing the contributions of individual membrane currents to somatic responses in spatially extended biophysical neuron models.
The demo visualizes how somatic activity is shaped by membrane currents in response to dendritic stimulation with varying numbers of synapses.
The core method recursively decomposes axial currents across neuronal compartments to attribute them to underlying membrane currents. The results are rendered as intuitive "currentscapes" — compact plots showing the dynamic contribution of ionic currents to neuronal output.

---


## Repository Elements


- `simulator/`: Contains the biophysical model simulator.
- `preprocessor/`: Extracts and cleans membrane and axial current data.
- `currentscape_calculator/`: Calculates the contributions of membrane currents.
- `currentscape_visualization/`: Plots currentscapes.
- `CurrentscapePipeline.py`: Core pipeline to run the simulation, preprocessing, currentscape calculation, and visualization.

---

## How to Run the Demo

### 1. **Install Dependencies**

Make sure you have Python 3.9 and `pip`. Then run:

```bash
pip install -r requirements.txt
```

---

### 2. **Run the Demo Script**

You can set the simulation parameters and launch the script from an IDE using `main.py`.

---

### 3. **Expected Output**

**Figures 3C, F from the paper can be recreated using this demo.**

- All outputs will be saved to an `output/` folder.
- Key files include:
  - `output/preprocessed/im.csv`, `iax.csv`: Preprocessed membrane and axial current data.
  - `output/results/part_pos.csv`, `part_neg.csv`: Current contributions.
  - `currentscape_Fig3C_caFalse_type_8.pdf`: Final currentscape plot.

This plot shows:
- The somatic membrane potential
- Total current flowing across the compartment
- The relative contribution of each membrane current to the neuronal activity over time

---

## Interpretation

The generated currentscape enables you to:
- Identify which dendritic regions and current types drive somatic responses

---
