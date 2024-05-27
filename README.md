# Signature Trading: 

A GitHub Repository for mean-variance Sig-Trading, corresponding to the paper:

**[Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4541830)**

- What are the main uses, why is it good?

## Requirements

## Functionality

## Practicalities to Consider

This repository provides a skeleton for fitting mean-variance optimal Signature Trading strategies. While some practicalities are essential and already included in this current framework (such as scaling the asset paths to start at 1, adding time and Hoff lead-lag transformation), there are many practicalities when working with financial data that will determine the performance of any given strategy, including (but not limited to):
-  Inputting signals that have predictability
-  Signal cleaning and ensuring signal paths do not start at zero
-  Tidying the sampling frequency irregularity of signals/assets, for online tasks keeping the signal value constant until the next update is advised
-  If the signal value is crucial, it may be important to
-  Time-add may be replaced by any other strictly monotone increasing coordinate (such as volume)
-  Final position scaling (vol-scaling, max-leverage constraints etc).

Many of these practicalities are pre-processing steps that can be amended within [sig_trader.py](src/sig_trader.py)

## Future Developement

A main source for improvement is computational efficiency and avoiding the curse of dimensionality as the time complexity for the fitting procedure is exponential in the number of dimensions and orders.

## License
