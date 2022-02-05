# PKL - Physics-aware Kernel Learning

## Main goal

We collected a wide range of methods for characterizing the concept of _persistence_ (memory, carry-over effects) in complex systems.

## Our survey paper

<b>Physics-aware Nonparametric Regression Models for Earth %and Climate Data Analysis</b><br>
Jordi Cort\'es-Andr\'es, Gustau Camps-Valls, Sebastian Sippel, Enik\H{o} Sz\'ekely,  Dino Sejdinovic, Emiliano Diaz, Adri\'an  P\'erez-Suay, Zhu Li, Miguel Mahecha, Markus Reichstein<br>
Environmental Research Letters

...
Abstract. Process understanding and modeling is at the core of scientific reasoning. Principled {\em parametric} and mechanistic modeling dominated science and engineering until the recent emergence of machine learning. Despite great success in many areas, machine learning algorithms in the Earth and climate sciences, and more broadly in physical sciences, are not explicitly designed to be physically-consistent and may, therefore, violate the most basic laws of physics. In this work, motivated by the field of algorithmic fairness, we reconcile data-driven machine learning with physics modeling by introducing a {\em nonparametric} and {\em nonlinear} physics-aware regression method. By incorporating a dependence-based regularizer, the method leads to models that are consistent with domain knowledge, as reflected by either simulations from physical models or ancillary data. The idea can conversely encourage independence of model predictions with other variables. The method is computationally efficient and comes with a closed-form analytic solution. Through a consistency-vs-accuracy path diagram, one can assess the consistency between data and physical models. We demonstrate through examples on simulations and real experiments in Earth and climate studies that, by incorporating a priori domain knowledge into the machine learning framework, it is possible to improve both the accuracy and the physical (domain-based) consistency of the machine learning model.
...

## Code & toolboxes

### Methods for short-term persistence characterization

```
Markov chain models for persistence characterization
Auto-regressive models: ARMA and ARIMA models
Persistence and memory effects in neural networks
    ARMA modeling with neural networks
    Synapses as digital filters
    Recurrent neural networks
Advanced memory-based neural networks
    Transformers and reformers
    Deep persistent memory network
    Augmented memory networks
Short-term persistence in non-stationary data streams
```

### Methods for long-term persistence characterization

```
Hurst Rescaled Range (R/S) analysis
Detrended Fluctuation Analysis (DFA)
Multi-fractal DFA version
Estimating the persistence strength (β) with the wavelet transform
```
## References
