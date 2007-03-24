function a = softmax2(n)
%SOFTMAX2     Softmax transfer function
%
%	Description
%	A = SOFTMAX2(N) takes a matrix N of network outputs and 
%       transfers it through a sotmax function to get A.
%       
%       Code:
%       temp = exp(n);
%       a = temp./(sum(temp, 2)*ones(1,size(n,2)));
%
%	See also
%	MLP2, MLP2PAK, MLP2UNPAK, MLP2ERR, MLP2BKP, MLP2GRAD
%

% Copyright (c) 1999 Aki Vehtari

temp = exp(n);
a = temp./(sum(temp, 2)*ones(1,size(n,2)));
