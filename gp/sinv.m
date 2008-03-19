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
    m = amd(A);
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
        ind{j} = I(inda{j})';
    end
%    inds=cellfun('prodofsize',ind);

    D = diag(L);
    L2 = L-sparse(1:n,1:n,D);
    Z = zeros(size(I));
    Z((I==n & J==n)) = 1./D(n).^2;
    Di = 1./D;
    % pre-allocate memory
    cindt=zeros(floor(n*n/2),1);
    cindit=zeros(n,1);
    for i=n-1:-1:1
        fil = full(L2(:,i)~=0 | Lb(:,i)~=0);
        fi = find(fil);
        l = L2(fi,i)';
        lfi=length(fi);
        
        %[cind, cindi] = ismember2c(fi, fil, inda, ind, cindt, cindit, i);
            i3=0;
            i4=0;
            for i1=1:lfi
                cind1=inda{fi(i1)};
                Icind1=ind{fi(i1)};
                for i2=1:numel(Icind1)
                    if fil(Icind1(i2))
                        i3=i3+1;
                        cindt(i3)=cind1(i2);
                    end
                    if Icind1(i2)==i
                        i4=i4+1;
                        cindit(i4)=cind1(i2);
                    end
                end
            end
            % remove extras
            cind=cindt(1:i3);
            cindi=cindit(1:i4);


        Zt = Z(cind);
        Zt=reshape(Zt,lfi,lfi);
        zij = -l*Zt.*Di(i);
        Z(cindi) = zij;
        indz = cumsum(histc(ind{i},[0 ; fi]));
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
    function [cind1, Icind1] = ismember2b(I,fi,inda, ind)    %, cind
        cind1=[]; Icind1 =[]; %cind=[];
        if ~isempty(fi)
            for kk=fi   %k=1:length(fi)
%                 h = (ind{kk});
%                 h2 = inda{kk};
%                 [c,ind_k,b] = intersect(h',fi);
                cind1 =[cind1 inda{kk}];
                Icind1 = [Icind1 ind{kk}];
%                cind = [cind h2(ind_k)];
            end
        cind1 = cind1';
        Icind1 = Icind1';
        %cind = cind';
        end
    end


end






% 
%     if nargin<2
%         B=A;
%     end
% 
%     m = symamd(A);
%     A = A(m,m);
%     B = B(m,m);
%     Lb =tril(B,-1);
%     L = chol(A, 'lower');
%     [I,J] = find(L+L'+B);
% 
% 
%     % Evaluate the sparse inverse
%     a1=zeros(n,1);
%     a2 = cumsum(histc(J,1:n));
%     a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;
% 
%     for j=1:n
%         inda{j} = a1(j):a2(j);
%         ind{j} = I(inda{j})';
%     end
% 
%     D = diag(L);
%     L2 = L-sparse(1:n,1:n,D);
%     Z = zeros(size(I));
%     Z((I==n & J==n)) = 1./D(n).^2;
%     Di = 1./D;
%     for i=n-1:-1:1
%         fi = find(L2(:,i)~=0 | Lb(:,i)~=0);
%         l = L2(fi,i)';
%         lfi=length(fi);
%         %cind1 = ismember2(J,fi,inda);
%         %Icind1 = I(cind1);
%         %Icind1 = ismember2(J,fi,ind);
%         [cind1, Icind1] = ismember2b(I,fi,inda, ind);
%         cind = cind1(ismembc(Icind1,fi)); %find(
%         %cind = cind1(logical(ismember2c(Icind1,fi,lfi))); %find(
%         %[cind1, Icind1, cind] = ismember2b(I,fi,inda, ind);
% 
%         Zt = Z(cind);
%         Zt=reshape(Zt,lfi,lfi);
%         zij = -l*Zt.*Di(i);
%         cind = cind1(Icind1==i);
%         Z(cind) = zij;
%         indz = cumsum(histc(ind{i},[0 ; fi]));  %I(a1(i):a2(i))
%         indz = a1(i) + indz(1:end-1);
%         Z(indz) = zij;
% 
%         zij = Di(i).^2-l*Z(indz).*Di(i);
%         Z(a1(i)-1+find(ind{i}==i,1)) = zij;
%     end
%     Z = sparse(I,J,Z);
%     r(m) = 1:n;
%     Z = Z(r,r);

