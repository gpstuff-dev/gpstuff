function [ct] = hct(crit,y,z,tt)
% HCT Calculate Harrel's C 
%
%   Description
%   [ct] = HCT(CRIT,Y,Z,TT)
%
%   Compute Harrel's C statistic estimate at time tt using 
%   criteria vector CRIT, observed time vector Y and event indicator vector Z
%   ( = 0 if event is experienced before tt and  =1 if not)

ip=inputParser;
ip.addRequired('crit',@(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y',@(x) ~isempty(x) && isreal(x))
ip.addRequired('z', @(x) ~isempty(x) && isreal(x))
ip.addRequired('tt', @(x) ~isempty(x) && isreal(x))

ip.parse(crit,y,z,tt)

    if size(y,2) ~= size(z,2)
       error('y and z dimensions must match')   
    end
   
  for i=1:size(tt,2)      
      comp=bsxfun(@and,bsxfun(@and,bsxfun(@lt,y,y'),y<=tt),z==0);
      conc=bsxfun(@gt,crit,crit').*comp;
      ct=sum(conc(:))./sum(comp(:));
  end
    
    
end

