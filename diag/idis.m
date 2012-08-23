function [id,bb,rt,rn] = idis(pt,pn,varargin)
% [ID,BB,RT,RN]= IDIS(PT,PN)
% 
% Description 
%
%   Given two vectors of cdf estimated probabilities (Pt = traditional model, Pn = new model)
%   returns Integrated Discrimination Improvement, it's Bayesian Bootstrap
%   estimated density, R² estimation of new and old model 
% 
%  OPTIONS
%      rsubstream    - number of a random stream to be used for
%                   simulating dirrand variables the data. This way
%                   same simulation can be obtained for different
%                   models. 

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

