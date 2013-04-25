% Monte Carlo FUNCTIONS (in the mc-folder):
%
% Markov chain Monte Carlo methods
%   GIBBS         - Gibbs sampling
%   HMC2          - Hybrid Monte Carlo sampling
%   HMC2_OPT      - Default options for Hybrid Monte Carlo sampling
%   HMC_NUTS      - No-U-Turn Sampler (NUTS)
%   METROP2       - Metropolis algorithm
%   METROP2_OPT   - Default options for Metropolis sampling
%   SLS           - Slice Sampling
%   SLS_OPT       - Default options for Slice Sampling
%   SLS1MM        - 1-dimensional fast minmax slice sampling
%   SLS1MM_OPT    - Default options for SLS1MM_OPT
%
% Monte Carlo methods
%   BBMEAN        - Bayesian bootstrap mean
%   BBPRCTILE     - Bayesian bootstrap percentile
%   RANDPICK      - Pick element from x randomly
%                   If x is matrix, pick row from x randomly.
%   RESAMPDET     - Deterministic resampling
%   RESAMPRES     - Residual resampling
%   RESAMPSIM     - Simple random resampling
%   RESAMPSTR     - Stratified resampling
%
% Manipulation of MCMC chains
%   THIN     - Delete burn-in and thin MCMC-chains
%   JOIN     - Join similar structures of arrays to one structure of arrays
%   TAKE_NTH - Take n'th sample from Monte Carlo structure
%   BATCHMC  - Batch MCMC sample chain and evaluate mean/median of batches
%
