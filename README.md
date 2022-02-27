# PKL - Physics-aware Kernel Learning

## Main goal

This repo accompanies the method physics-aware kernel learning (PKL) for nonparametric regression. 

## Our survey paper

<b>Physics-aware Nonparametric Regression Models for Earth Data Analysis</b><br>
Jordi Cortés-Andrés, Gustau Camps-Valls, Sebastian Sippel, Enikó Székely, Dino Sejdinovic, Emiliano Diaz, Adrián  Pérez-Suay, Zhu Li, Miguel Mahecha, Markus Reichstein<br>
Environmental Research Letters, 2022

**Abstract** Process understanding and modeling is at the core of scientific reasoning.Principled parametric and mechanistic modeling dominated science and engineering until the recent emergence of machine learning. Despite great success in many areas, machine learning algorithms in the Earth and climate sciences, and more broadly in physical sciences, are not explicitly designed to be physically-consistent and may, therefore, violate the most basic laws of physics. In this work, motivated by the field of algorithmic fairness, we reconcile data-driven machine learning with physics modeling by illustrating a nonparametric and nonlinear physics-aware regression method. By incorporating a dependence-based regularizer, the method leads to models that are consistent with domain knowledge, as reflected by either simulations from physical models or ancillary data. The idea can conversely encourage independence of model predictions with other variables that are known to be uncertain either in their representation or magnitude. The method is computationally efficient and comes with a closed-form analytic solution. Through a consistency-vs-accuracy path diagram,one can assess the consistency between data-driven models and physical models. Wedemonstrate through three examples on simulations and measurement data in Earth and climate studies that, by incorporating a priori domain knowledge into the machine learning framework, it is possible to improve on the trade-off between accuracy and the physical (domain-based) consistency of the machine learning
model.

## Code & data

  * Physics-aware kernel regression (in MATLAB)
  * Fair/consistent Gaussian process regression (in Python)
  * Demos
  * Datasets

## References


