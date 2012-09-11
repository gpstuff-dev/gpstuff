function [id,bb,rt,rn] = idis(pt,pn,varargin)
%IDIS Integrated Discrimination Improvement given probabilities from two models
% 
%  Description 
%    IDI = IDIS(PT,PN,OPTIONS) Returns Integrated Discrimination Improvement (IDI)
%    given two vectors of probabilities: PT for traditional model
%    and PN for new model.
%
%    [IDI,BB] = IDIS(PT,PN,OPTIONS) Returns also Bayesian bootstrap
%    samples BB from the distribution of the IDI statistic.
%
%    [IDI,BB,RT,RN] = IDIS(PT,PN,OPTIONS) Returns also R^2
%    statistics for two models: RT for traditional model and RN for
%    new model.
%   
%    OPTIONS is optional parameter-value pair
%      rsubstream - number of a random stream to be used for
%                   simulating dirrand variables. This way same
%                   simulation can be obtained for different models. 
%                   See doc RandStream for more information.
%
%  Reference
%    L. E. Chambless, C. P. Cummiskey, and G. Cui (2011). Several
%    methods to assess improvement in risk prediction models:
%    Extension to survival analysis. Statistics in Medicine
%    30(1):22-38.

ip=inputParser;
ip.addRequired('pt',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('pn',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('rsubstream', 0, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
ip.parse(pt,pn,varargin{:})
rsubstream=ip.Results.rsubstream;

if nargin<3
  [rt,bbt]=rsqr(pt);
  [rn,bbn]=rsqr(pn);
else
  [rt,bbt]=rsqr(pt,'rsubstream',rsubstream);
  [rn,bbn]=rsqr(pn,'rsubstream',rsubstream);
end
id=rn-rt;
bb=bbn-bbt;
end
