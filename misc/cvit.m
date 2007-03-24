function [itr, itst] = cvit(n, k)
% CVIT - Create itr and itst indeces for k-fold-cv
%   

%   Author: Aki Vehtari <Aki.Vehtari@hut.fi>
%   Last modified: 2006-12-21 13:29:48 EET

if nargin < 2
  k=10;
end
a=k-rem(n,k);
b=floor(n/k);
for cvi=1:a
  itst{cvi}=[[1:b]+(cvi-1)*b]; 
  itr{cvi}=setdiff(1:n,itst{cvi}); 
end
for cvi=(a+1):k
  itst{cvi}=[(a*b)+[1:(b+1)]+(cvi-a-1)*(b+1)]; 
  itr{cvi}=setdiff(1:n,itst{cvi}); 
end  
