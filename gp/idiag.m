function iD = idiag2(A,B)
% IDIAG    Evaluate the diagonal of inverse matrix
%
% iD = idiag(A)  returns diag(inv(A)). In case of sparse matrix
%                iD is evaluated without forming the whole inverse
%                matrix. See Rue and ... for details.
%
% iD = idiag(A,B)  returns diag(inv(A)*B)
%
%   Note! The function works only for symmetric matrices!

% Copyright (c) 2007      Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.
n = size(A,1);

if n<100
    % The implementation with full matrix
    % Handling the sparse matrix storage system gives overhead
    % in computation time which can be prevented using full matrix storage
    % system if it is possible without memory problems

    if nargin==1
        m = amd(A);
        A = A(m,m);
        L = lchol(A);
        D = diag(L);
        L2 = L-sparse(1:n,1:n,D);
        Z = zeros(n,n);
        for i=n:-1:1
            j = find(L2(:,i)~=0);
            zij = - L2(j,i)'*Z(j,j)./D(i);
            Z(i,j) = zij;
            Z(j,i) = zij;
            Z(i,i)=1./D(i).^2 - L2(j,i)'*Z(j,i)./D(i);
        end
        r(m) = 1:n;
        iD = diag(Z(r,r));

    elseif nargin == 2

        m = symamd(A);
        A = A(m,m);
        B = B(m,m);
        Lb =tril(B,-1);
        L = lchol(A);
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
        iD = sum(Z.*B',2);
        iD = iD(r);
    else
        error('Wrong number of arguments in! \n')
    end

else

    
    [LD, p, q] = ldlchol(A);
    [I,J,ld] = find(LD);
    temp = [I(:) J(:) ; J(:) I(:)];
    temp = sortrows(unique(temp,'rows'),2);
    Iz = temp(:,1); Jz = temp(:,2); 
    
    % Find the column starting points
    a1=zeros(n,1);
    a2 = cumsum(histc(J,1:n));
    a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;
    az1=zeros(n,1);
    az2 = cumsum(histc(Jz,1:n));
    az1(1) = 1; az1(2:end) = az2(1:end-1) + 1;
    
    for j=1:n
        indaz{j} = az1(j):az2(j);
        indIz{j} = Iz(indaz{j})';
    end
    
    % Evaluate the sparse inverse
    z = zeros(size(Iz));
    z(end) = 1./ld(end);
    % Allocate memory
    cindit=zeros(n,1);
    for jj = n-1:-1:1
        fil = ld(a1(jj)+1:a1(jj+1)-1);
        fi = I(a1(jj)+1:a1(jj+1)-1);
        lfi = length(fi);
        Zt = zeros(lfi,lfi);
        indz = cumsum(histc(indIz{jj},[0 ; fi]));
        indz = az1(jj) + indz(1:end-1);
        i4=0;            
        for i1 = 1:lfi
            cind1=indaz{fi(i1)};
            Icind1=indIz{fi(i1)};
            indfi = 1;
            i3=0;
            for i2=1:length(Icind1)
                if Icind1(i2)==jj  % Find the indeces for the jj'th rows in fi columns
                    i4=i4+1;
                    cindit(i4)=cind1(i2);
                end
                if indfi <= lfi && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
                    i3 = i3 + 1;
                    Zt(i3,i1) = z(cind1(i2));
                    indfi = indfi+1;
                end
            end
        end
        % remove extras
        cindi=cindit(1:i4);
        
        zij = -fil'*Zt;
        z(cindi) = zij;
        z(indz) = zij;
        zij = 1./ld(a1(jj)) - fil'*z(indz);
        z(az1(jj)-1+find(indIz{jj}==jj,1)) = zij;
    end
    
    if nargin == 1
        Z = sparse(Iz,Jz,z);
        r(q) = 1:n;
        iD = diag(Z);
        iD = iD(r);
        
    elseif nargin == 2
        Z = sparse(Iz,Jz,z);
        r(q) = 1:n;
        Z = Z(r,r);
        iD = sum(Z.*B',2);

    else
        error('Wrong number of arguments in! \n')
    end

end

    function ind = ismember2(J,fi,inda)
        ind=[];
        for kk=fi
            ind =[ind inda{kk}];
        end
        ind = ind';
    end


end






















% find the set of needed elements
%     C = L + L';
%     [I,J]=find(C);
%     change = 1;
%     while change == 1
%         for i=n:-1:1
%             %a1 = find(J==i,1);
%             %fi = I(a1-1+find(I(a1:end)>i, 1));
%             fi = find(J==i);
%             fi = fi(1)-1 + find(I(fi)>i);
%             k = I(fi);
%             C(k,k) = 1;
%         end
%         if nnz(C)==length(I)
%            change = 0;
%         end
%         [I,J] = find(C+C');
%     end












%
%
% L = chol(A, 'lower');
% [I,J,V]=find(L+L');
% D = diag(L);
%
% Z = zeros(size(V));
%
% for i=n:-1:1
%     fi = find(I>i & J==i);
%     k = I(fi);
%     fs = zeros(size(k));
%     for j=n:-1:i
%         for kk=1:length(k)
%             ind = find(I==k(kk) & J==j);
%             if isempty(ind)
%                 I = [I ; k(kk)];
%                 J = [J ; j];
%                 Z = [Z ; 0];
%                 fs(kk) = length(I);
%             else
%                 fs(kk) = ind;
%             end
%         end
%         zij =1./D(i).^2*(i==j) - V(fi)'*Z(fs)./D(i);
%         Z(find(I==i & J==j)) = zij;
%         Z(find(I==j & J==i)) = zij;
%     end
% end
% iD = diag(sparse(I,J,Z,n,n));


% L = chol(A, 'lower');
% [I,J,V]=find(L+L');
% D = diag(L);
%
% Z = spalloc(n,n,length(I));
% for i=n:-1:1
%     fi = find(I>i & J==i);
%     k = I(fi);
%     for jj=length(k):-1:1 %j=n:-1:i
%         j = k(jj);
%         for kk=1:length(k)
%             ind = find(I==k(kk) & J==j);
%             if isempty(ind)
%                 I = [I ; k(kk)];
%                 J = [J ; j];
%                 Z = [Z ; 0];
%                 fs(kk) = length(I);
%             else
%                 fs(kk) = ind;
%             end
%         end
%         zij =1./D(i).^2*(i==j) - V(fi)'*Z(fs)./D(i);
%         Z(find(I==i & J==j)) = zij;
%         Z(find(I==j & J==i)) = zij;
%     end
% end
%

%
%     % find the set of needed elements
%     L = chol(A, 'lower');
%     [I,J,V]=find(L);
%     C =L;
%     for i=n:-1:1
%         fi = find(I>i & J==i);
%         k = I(fi);
%         for jj=length(fi):-1:1
%             j=k(jj);
%             C(k,j)=1;
%         end
%     end
%     [I,J] = find(C);
%
%



%     iD = diag(Z);
%
%
%
%     L = chol(A, 'lower');
%     [I,J,V]=find(L+L');
%
%     % Find the set of nodes
%     %adI = zeros(length(I),1); adJ=zeros(length(I),1);
%     aa=1;
%     for i=n:-1:1
%         fi = I(find(I>i & J==i));
%        % fi = find(I>i & J==i);
%         for jj = length(fi):-1:1
%             j=fi(jj);
%             for kk=1:length(fi)
%                 k = fi(kk);
%                 if  isempty(find(I==j & J==k))
%                     adI(aa) = j;
%                     adJ(aa) = k;
%                     aa = aa+1;
%                 end
%             end
%         end
%     end
%     I = [I ; adI];
%     J = [J ; adJ];
%     [I,J] = find(sparse(I,J,1));
%     % Evaluate the needed inverse matrix elements






%         m = symamd(A);
%         A = A(m,m);
%         L = chol(A, 'lower');
%         D = diag(L);
%         L2 = L-sparse(1:n,1:n,D);
%         [I,J,V]=find(L2);
%
%         [Iz,Jz]=find(L+L');
%         Z = zeros(length(Iz),1);
%
%         Z(find(Iz==n & Jz==n))=1./D(n).^2;
%         for i=n-1:-1:1
%             fi = (J==i);
%             j = I(fi);
%             fs = (Jz==j);
%
%             zij = -V(fi)'*zz./D(i);
%
%             Z(find(Iz==i & Jz==j)) = zij;
%             Z(find(Iz==j & Jz==i)) = zij;
%             for jj=fi(length(fi):-1:1) %j = I(fi(length(fi):-1:1))
%                 j=I(jj);
%                 zz=zeros(length(fi),1);
%                 s=1;
%                 for kk=fi  %k = I(fi)
%                     k = I(kk);
%                     zz(s) = Z(find(Iz==k & Jz==j));
%                     s=s+1;
%                 end
%
%             end
%
%             fs=zeros(1,length(fi));
%             s=1;
%             for kk=fi  %k = I(fi)
%                 k = I(kk);
%                 fs(s) = find(Iz==k & Jz==i);
%                 s=s+1;
%             end
%             fs = find(Iz==i & Jz==i);
%             Z(find(Iz==i & Jz==i))=1./D(i).^2 - sum(V(fi).*Z(fs))./D(i);
%         end
%         r(m) = 1:n;
%         iD = diag(Z);








%                 % Evaluate the sparse inverse
%                 a1=zeros(n,1);
%                 a2 = cumsum(histc(J,1:n));
%                 a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;
%                 %        a2 = a1(2:end)-1; a2(n)=length(J);
%         
%                 for j=1:n
%                     inda{j} = a1(j):a2(j);
%                     ind{j} = I(inda{j});
%                 end
%         
%                 D = diag(L);
%                 L2 = L-sparse(1:n,1:n,D);
%                 Z = zeros(size(I));
%                 Z(find(I==n & J==n)) = 1./D(n).^2;
%                 for i=n-1:-1:1
%                     fi = [i ; find(L2(:,i)~=0)];
%                     %indz = cumsum(histc( I(inda{j}),[0 ; fi]));
%                     %indz = flipud(a1(i) + indz(1:end-1));
%                     for jj = [length(fi):-1:1]
%                         j = fi(jj);
%                         %ind = I(a1(j):a2(j));
%                         zij = - L2(ind{j},i)'*Z(inda{j})./D(i)  + 1./D(i).^2*(i==j);
%         
%                         %Z(find(I==i & J==j,1)) = zij + 1./D(i).^2*(i==j);
%                         %Z(find(I==j & J==i,1)) = zij + 1./D(i).^2*(i==j);
%                         Z(a1(j)-1+find(ind{j}==i,1)) = zij;
%                         Z(a1(i)-1+find(I(inda{i})==j,1)) = zij;
%                         %Z(indz(jj)) = zij;
%                     end
%                 end
%                 Z = sparse(I,J,Z);
%                 r(m) = 1:n;
%                 iD = diag(Z(r,r));
