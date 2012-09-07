function [r,bb] = rsqr(p,varargin)
%RSQR R^2 statistic given probabilities
%
%  Description:
%    R = RSQR(P) Returns R^2 statistic given vector P of estimated
%    probabilities of presenting event before time T and using
%    Bayesian bootstrap its density estimate
%
%    [R,BB] = RSQR(P) Returns also Bayesian bootstrap samples BB
%    from the distribution of the R^2 statistic.
%
%  Options
%    rsubstream - number of a random stream to be used for
%                 simulating dirrand variables. This way same
%                 simulation can be obtained for different models. 
%                 See doc RandStream for more information.
%
%  Reference
%    L. E. Chambless, C. P. Cummiskey, and G. Cui (2011). Several
%    methods to assess improvement in risk prediction models:
%    Extension to survival analysis. Statistics in Medicine
%    30(1):22-38.
%

% Copyright (C) 2012 Ernesto Ulloa, Aki Vehtari

ip=inputParser;
ip.addRequired('p',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('rsubstream', 0, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
ip.parse(p,varargin{:})
rsubstream=ip.Results.rsubstream; 

s=1-p;
Es=mean(s);
Vs=mean(s.^2)-Es^2;
r=Vs/(Es*(1-Es));

if nargout>1

  if rsubstream>0
    stream = RandStream('mrg32k3a');
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      prevstream=RandStream.setDefaultStream(stream);
    else
      prevstream=RandStream.setGlobalStream(stream);
    end
    stream.Substream = rsubstream;
  end
  
  n=1000;
  qr=dirrand(size(p,1),n);

  for i=1:n
    Esr=wmean(s,qr(:,i));
    Vsr=wmean(s.^2,qr(:,i))-Esr^2;
    bb(i,1)=Vsr/(Esr*(1-Esr));    
  end

  if rsubstream>0
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      RandStream.setDefaultStream(prevstream);
    else
      RandStream.setGlobalStream(prevstream);
    end
  end
  
end 
end

