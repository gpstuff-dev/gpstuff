function [itr, itst] = cvitr(n, k)
% CVIT - Create itr and itst indeces for k-fold-cv with ranndom permutation
%   

%   Author: Aki Vehtari <Aki.Vehtari@hut.fi>
%   Last modified: 2007-01-26 16:33:10 EET

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
rii=randperm(n);
for cvi=1:k
  itst{cvi}=rii(itst{cvi});
  itr{cvi}=rii(itr{cvi});
end
