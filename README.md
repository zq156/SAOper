# SAOper

Note: This repository is for the paper SAOper.

## Installation

-   **Environment Setup**: First, clone this repository and create the Conda environment:
    ```bash
    git clone https://github.com/zq156/SAOper.git
    cd SAOper
    conda create -n saoper python=3.10 -y
    ```
-   **Dependencies**: Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

We use the **D4RL** **Adroit Hand** dataset for experiments.
-   Download the dataset from minari.
-   The experiments are conducted in the MuJoCo simulation environment.

## Usage

### Reward model Training

To train the reward model:
1.  Modify the training parameters in `configs/samples/agents/adroit_door.yml` if necessary.
2.  Run the IRL training script:
    python irl_methods/irl_samples_ml_irl_SAOper.py

### RL Training

To a train varying goal tasks, use the RL training script:
python irl_methods/rl_SAOper.py