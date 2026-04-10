# Custom LSTM 🧠

A research project exploring the theoretical mathematics behind Long Short-Term Memory (LSTM) networks, with the goal of understanding their inner workings from first principles and investigating potential improvements to the standard architecture.

> **Note:** A detailed write-up is available in [`README.pdf`](./README.pdf) in the root of this repository.

---

## Motivation

Standard deep learning frameworks abstract away the internals of LSTM cells, making it difficult to study or experiment with the underlying math. This project implements an LSTM from scratch — without relying on high-level libraries like Keras or PyTorch's built-in LSTM layers — in order to:

- Deeply understand the gate equations (input, forget, output, and cell state)
- Verify correctness against established implementations
- Explore modifications to the standard LSTM formulation and measure their impact

---

## Project Structure

```
Custom-LSTM/
├── Code/               # Jupyter notebooks implementing the custom LSTM
├── Documents/          # Supporting documents and write-ups
├── Images/             # Figures, diagrams, and result visualizations
├── Literature/         # Reference papers and academic sources
├── Proposal/           # Original project proposal
└── README.pdf          # Detailed project report
```

---

## Background: What is an LSTM?

An LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) designed to learn long-term dependencies in sequential data. Standard RNNs struggle with the **vanishing gradient problem** — gradients shrink to near zero during backpropagation through many time steps, preventing the network from learning long-range patterns.

LSTMs solve this with a **cell state** — a memory that runs through the sequence — regulated by three learned gates:

| Gate | Purpose |
|------|---------|
| **Forget gate** | Decides what information to discard from the cell state |
| **Input gate** | Decides what new information to write to the cell state |
| **Output gate** | Decides what part of the cell state to output as the hidden state |

The core equations at each time step `t` are:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)       # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)       # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)   # Candidate cell state
C_t = f_t * C_{t-1} + i_t * C̃_t          # Updated cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)       # Output gate
h_t = o_t * tanh(C_t)                      # Hidden state output
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Preston-Robertson/Custom-LSTM.git
cd Custom-LSTM
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

*(If no `requirements.txt` is present, the core dependencies are `numpy`, `matplotlib`, and `jupyter`.)*

**3. Launch Jupyter**

```bash
jupyter notebook
```

Then open the notebooks in the `Code/` folder.

---

## Repository Contents

### `Code/`
Jupyter notebooks containing the custom LSTM implementation, experiments, and comparisons. The notebooks walk through:
- Forward pass implementation
- Backpropagation through time (BPTT)
- Training and evaluation on test sequences
- Comparisons with standard LSTM behavior

### `Documents/`
Supporting write-ups and analysis documents related to the project.

### `Images/`
Visualizations generated during experimentation, including training curves, gate activation plots, and architecture diagrams.

### `Literature/`
Academic papers and references that informed the project, including foundational LSTM literature.

### `Proposal/`
The original project proposal outlining the research goals and methodology.

---

## References

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to Forget: Continual Prediction with LSTM*.
- Olah, C. (2015). *Understanding LSTM Networks*. [colah.github.io](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## License

This project does not currently specify a license. All rights reserved by the author.
