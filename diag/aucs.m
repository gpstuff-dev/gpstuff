function [a,fps,tps] = aucs(crit,z)
% [A,FPS,TPS] = AUCS(CRIT,Z)
%
% Given criteria vector and censoring indicator column vector (0=exact, 1=censored) 
% returns AUC, False Positives and True Positives vectors 
%
%
% NOTE: If criteria vector is Eft be carefull when using likelihoods as, for example
% log-logistic since criteria vector is sorted using 'descend'.

ip=inputParser;
ip.addRequired('crit',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('z', @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(crit,z)
 
 ye=z;
 tps=0;
 fps=0;
 qs=sort(crit,'descend');
 

 
 for i2=1:numel(crit)
     tps(i2+1)=mean(crit>=qs(i2) & 1-ye)./mean(1-ye);
     fps(i2+1)=mean(crit>=qs(i2) & ye)./(mean(ye));
 end
 
  if (mean(1-ye)==0)||(mean(ye)==0) 
    warning('z vector has no different values, function will return 0');
    a=0;
  else
    a=sum([diff(fps).*mean([tps(1:end-1);tps(2:end)],1)]);  
  end

end

