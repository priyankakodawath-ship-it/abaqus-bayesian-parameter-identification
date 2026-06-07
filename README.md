# Abaqus Bayesian Parameter Identification

## Overview

This project demonstrates an automated Python workflow for inverse parameter identification using Abaqus simulations and Bayesian Optimization.

The framework automatically:

- Updates Abaqus input files (.inp)
- Runs Abaqus simulations
- Extracts force-displacement responses
- Compares simulation results with experimental data
- Calculates RMSE
- Uses Bayesian Optimization with Gaussian Process Regression
- Identifies optimal material parameters

## Parameters Identified

- Young's Modulus (E)
- Cohesive Stiffness (k)
- Maximum Stress (Smax)
- Damage Evolution Parameter (dmax)
- Friction Coefficient (μ)

## Technologies

- Python
- Abaqus
- NumPy
- SciPy
- Scikit-Learn
- Gaussian Process Regression
- Bayesian Optimization

## Engineering Skills Demonstrated

- Finite Element Analysis (FEA)
- Abaqus Automation
- Material Parameter Calibration
- Optimization
- Python Scripting
- Data Processing
- Force-Displacement Curve Matching

## Workflow

Experimental Data

↓

Modify Abaqus Input File

↓

Run Abaqus Simulation

↓

Extract FE Results

↓

Calculate RMSE

↓

Bayesian Optimization

↓

Update Parameters

↓

Repeat Until Convergence

## Author

Priyanka parameshwaranaik

M.Sc. Aerospace Engineering

Technical University of Munich
