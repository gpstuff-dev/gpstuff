function [at] = auct(crit,y,z,tt)
%   [AT] = ASSESS(CRIT,Y,Z,TT)
%
% Description
% Given criteria vector CRIT, observed time vector Y, censoring indicator matrix Z (or vector)
% at time vector tt (0=exact, 1=censored) 
% and TT time vector (or single value) this function returns vector AUC(TT) evaluated 
% in each element of TT. 
%
%
% %NOTE: Be carefull when using likelihoods as, for example, log-logistic since 
% criteria vector will be compared using @gt and not @lt'.
ip=inputParser;
ip.addRequired('crit',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) isreal(x) && all(isfinite(x(:))))
ip.addRequired('z', @(x) isreal(x) && all(isfinite(x(:))))
ip.addRequired('tt', @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(crit,y,z,tt)
  
 for i=1:size(tt,2)
 comp=bsxfun(@times,bsxfun(@and,y<=tt(i),1-z(:,i)),bsxfun(@or,bsxfun(@and,y<=tt(i),z(:,i)),y>=tt(i))');
 conc=bsxfun(@times,bsxfun(@gt,crit,crit'),comp);
 at(i)=sum(conc(:))./sum(comp(:));
 end

end

