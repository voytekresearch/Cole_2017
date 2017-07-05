# Cole_2017
Analysis code ran to create the figures in [Cole et al., 2017, J Neuro](http://www.jneurosci.org/content/37/18/4830), entitled "Nonsinusoidal Beta Oscillations Reflect Cortical Pathophysiology in Parkinson's Disease."

## Notebooks

This repo contains one notebook per figure, showing the computation for each panel.

## Libraries

**util.py** - functions for loading data, running analysis across all subjects, and other miscellaneous tools

**shape.py** - functions for quantifying the waveform shape of the oscillations. See note below.

**pac.py** - algorithms for estimating phase-amplitude coupling

**plt.py** - functions for plotting results

## Quantifying oscillation shape

For future studies characterizing the waveform shape of oscillations, see the actively maintained `shape` module in [neurodsp](https://github.com/voytekresearch/neurodsp).
