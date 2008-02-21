function Z = sinv(A,B)
% SINV    Evaluate the sparse inverse matrix
%
% z = sinv(A,B)  returns the elements of inv(A)_ij, for which B_ij
%      is different from zero. See Rue and ... for details.
%
% z = sinv(A)  sets B=A and computes the same as sinv(A,B).
%
%   Note! The function works only for symmetric matrices!

% Copyright (c) 2007      Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.
n = size(A,1);

if n<1000
    % The implementation with full matrix
    % Handling the sparse matrix storage system gives overhead
    % in computation time which can be prevented using full matrix storage
    % system if it is possible without memory problems

    if nargin<2
        B=A;
    end
    m = symamd(A);
    A = A(m,m);
    B = B(m,m);
    Lb =tril(B,-1);
    L = chol(A, 'lower');
    D = diag(L);
    L2 = L-sparse(1:n,1:n,D);
    Z = zeros(n,n);
    for i=n:-1:1
        j = find(L2(:,i)~=0 | Lb(:,i)~=0);
        zij = - L2(j,i)'*Z(j,j)./D(i);
        Z(i,j) = zij;
        Z(j,i) = zij;
        Z(i,i)=1./D(i).^2 - L2(j,i)'*Z(j,i)./D(i);
    end
    r(m) = 1:n;
    Z = Z(r,r);

else

    if nargin<2
        B=A;
    end

    m = symamd(A);
    A = A(m,m);
    B = B(m,m);
    Lb =tril(B,-1);
    L = chol(A, 'lower');
    [I,J] = find(L+L'+B);


    % Evaluate the sparse inverse
    a1=zeros(n,1);
    a2 = cumsum(histc(J,1:n));
    a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;

    for j=1:n
        inda{j} = a1(j):a2(j);
        ind{j} = I(inda{j});
    end

    D = diag(L);
    L2 = L-sparse(1:n,1:n,D);
    Z = zeros(size(I));
    Z((I==n & J==n)) = 1./D(n).^2;
    Di = 1./D;
    for i=n-1:-1:1
        fi = find(L2(:,i)~=0 | Lb(:,i)~=0);
        l = L2(fi,i)';
        lfi=length(fi);
        cind1 = ismember2(J,fi,inda);
        Icind1 = I(cind1);
        cind = cind1(find(ismembc(Icind1,fi)));

        Zt = Z(cind);
        Zt=reshape(Zt,lfi,lfi);
        zij = -l*Zt.*Di(i);
        cind = cind1(Icind1==i);
        Z(cind) = zij;
        indz = cumsum(histc( I(a1(i):a2(i)),[0 ; fi]));
        indz = a1(i) + indz(1:end-1);
        Z(indz) = zij;

        zij = Di(i).^2-l*Z(indz).*Di(i);
        Z(a1(i)-1+find(ind{i}==i,1)) = zij;
    end
    Z = sparse(I,J,Z);
    r(m) = 1:n;
    Z = Z(r,r);

end

    function ind = ismember2(J,fi,inda)
        ind=[];
        for kk=fi
            ind =[ind inda{kk}];
        end
        ind = ind';
    end


end

