function [r,bb] = rsqr(p,varargin)
% [R,BB]=rsqr(P,OPTIONS)
%
% Description:
%   r = rsqr(p) Given p vector of estimated probabilities of presenting 
%   event before time t returns r² statistic and using bayseian bootstrap
%   its density estimate
%
% OPTIONS
%      rsubstream    - number of a random stream to be used for
%                   simulating dirrand variables the data. This way
%                   same simulation can be obtained for different
%                   models. 


ip=inputParser;
ip.addRequired('p',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('rsubstream', 0, @(x) isreal(x) && isscalar(x) && isfinite(x) && x>0)
ip.parse(p,varargin{:})
rsubstream=ip.Results.rsubstream; 


s=1-p;
Es=mean(s);
Vs=mean(s.^2)-Es^2;
r=Vs/(Es*(1-Es));

if nargin<2
    n=2000;
    qr=dirrand(size(p,1),n);

    for i=1:n
        Esr=wmean(s,qr(:,i));
        Vsr=wmean(s.^2,qr(:,i))-Esr^2;
        bb(i,1)=Vsr/(Esr*(1-Esr));    
    end
end
  
  if rsubstream>0
 
      stream = RandStream('mrg32k3a');
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
        prevstream=RandStream.setDefaultStream(stream);
    else
        prevstream=RandStream.setGlobalStream(stream);
    end
        stream.Substream = rsubstream;
   
    n=1500;
    qr=dirrand(size(p,1),n);

        for i=1:n
            Esr=wmean(s,qr(:,i));
            Vsr=wmean(s.^2,qr(:,i))-Esr^2;
            bb(i,1)=Vsr/(Esr*(1-Esr));    
        end

        if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
            RandStream.setDefaultStream(prevstream);
        else
            RandStream.setGlobalStream(prevstream);
        end;
  
  end 
end

