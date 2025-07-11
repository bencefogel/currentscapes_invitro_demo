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

Running the whole pipeline takes approximately 10 minutes.

### 1. **Install Dependencies**

Make sure you have Python 3.9 and `pip`. Then run:

```bash
pip install -r requirements.txt
```

---

### 2. **Compile .mod files**

Before running simulations, you need to compile NEURON's `.mod` files.  
Navigate to the folder containing the `.mod` files:

```bash
cd currentscapes_demo/simulator/model/density_mechs
nrnivmodl
```

This will compile all mechanism files required for the biophysical model.  

**Note**:
In some cases, NEURON may not find the compiled .mod files.
This can be fixed by copying the .mod files into the main project directory (e.g., currentscapes_demo/) and running nrnivmodl again there.
  
More information about NEURON: https://neuron.yale.edu/neuron<br>
More information about working with .mod files: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3263<br>
More information about compiling .mod files: https://nrn.readthedocs.io/en/latest/guide/faq.html#how-do-i-compile-mod-files

---

### 3. **Run the Demo Script**

You can set the simulation parameters and launch the script from an IDE using `main.py`.

If the currentscape results are already calculated and present in the output directory, the pipeline will skip the calculation steps and only run the visualization. This allows you to quickly regenerate plots without rerunning the entire pipeline.

---

### 4. **Expected Output**

**Figures 3C, F from the paper can be recreated using this demo.**

- All outputs will be saved to an `output/` folder.
- Key files include:
  - `output/preprocessed/im.csv`, `iax.csv`: Preprocessed membrane and axial current data.
  - `output/results/part_pos.csv`, `part_neg.csv`: Current contributions.
  - `currentscape_Fig3C_caFalse_type_8.pdf`: Final currentscape plot.

The currentscape plot shows:
- The somatic membrane potential
- Total current flowing across the compartment
- The relative contribution of each membrane current to the neuronal activity over time

---

## Interpretation

The generated currentscape enables you to:
- Identify which dendritic regions and current types drive somatic responses

---
