function x=cholrankup(R,U,V,b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Solve the system Ax=b, where is given in terms of its Cholesky
% factors plus a nonsymmetric rank r matrix
%
% A=R'*R+U*V'
%
% The matrix R is upper triangular n by n. The matrices, U and V, are 
% n by r.
%
% U=[u1,u2,...,ur];
% V=[v1,v2,...,vr];
%
% The vector b is the n by 1 right hand side.
%
% Written by: Greg von Winckel - 08/13/05
% Contact: gregvw(at)chtm(at)unm(at)edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[n,r]=size(V); I=eye(r);
temp=(R\(R'\[U b]));   % n x r+1
W=temp(:,1:r);         % n x r 
c=temp(:,r+1);         % n x 1
M=V'*temp;             % r x r+1
a=(M(1:r,1:r)+I)\M(:,r+1);  % r x 1
x=c-W*a;
