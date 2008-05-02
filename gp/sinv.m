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
        Z = Z(r,r);

    else

        [LD, p, q] = ldlchol(A);
        [I,J,ld] = find(LD);
        [Iz,Jz] = find(LD+LD');
% $$$         temp = [I(:) J(:) ; J(:) I(:)];
% $$$         temp = sortrows(unique(temp,'rows'),2);
% $$$         Iz = temp(:,1); Jz = temp(:,2); 
        
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
                indfi = lfi;
                i2=length(Icind1);
                go = true;
                while go
                    if Icind1(i2)==jj  % Find the indeces for the jj'th rows in fi columns
                        i4=i4+1;
                        cindit(i4)=cind1(i2);
                        go = false;
                    end
                    if indfi >= 1 && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
                        Zt(indfi,i1) = z(cind1(i2));
                        indfi = indfi-1;
                    end
                    i2 = i2-1;
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
        
        Z = sparse(Iz,Jz,z);
        r(q) = 1:n;
        Z = Z(r,r);
        

    end
        
end




% $$$             i4=0;            
% $$$             for i1 = 1:lfi
% $$$                 cind1=indaz{fi(i1)};
% $$$                 Icind1=indIz{fi(i1)};
% $$$                 indfi = 1;
% $$$                 for i2=1:length(Icind1)
% $$$                     if Icind1(i2)==jj  % Find the indeces for the jj'th rows in fi columns
% $$$                         i4=i4+1;
% $$$                         cindit(i4)=cind1(i2);
% $$$                     end
% $$$                     if indfi <= lfi && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
% $$$                         Zt(indfi,i1) = z(cind1(i2));
% $$$                         indfi = indfi+1;
% $$$                     end
% $$$                 end
% $$$             end
% $$$             % remove extras
% $$$             cindi=cindit(1:i4);




% $$$         [LD, p, q] = ldlchol(A);
% $$$         [I,J,ld] = find(LD);
% $$$         temp = [I(:) J(:) ; J(:) I(:)];
% $$$         temp = sortrows(unique(temp,'rows'),2);
% $$$         Iz = temp(:,1); Jz = temp(:,2); 
% $$$         
% $$$         % Find the column starting points
% $$$         a1=zeros(n,1);
% $$$         a2 = cumsum(histc(J,1:n));
% $$$         a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;
% $$$         az1=zeros(n,1);
% $$$         az2 = cumsum(histc(Jz,1:n));
% $$$         az1(1) = 1; az1(2:end) = az2(1:end-1) + 1;
% $$$         
% $$$         for j=1:n
% $$$             indaz{j} = az1(j):az2(j);
% $$$             indIz{j} = Iz(indaz{j})';
% $$$         end
% $$$ 
% $$$         % Evaluate the sparse inverse
% $$$         z = zeros(size(Iz));
% $$$         z(end) = 1./ld(end);
% $$$         % Allocate memory
% $$$         cindt=zeros(floor(n*n/2),1);
% $$$         cindit=zeros(n,1);
% $$$         for jj = n-1:-1:1
% $$$             fil = ld(a1(jj)+1:a1(jj+1)-1);
% $$$             fi = I(a1(jj)+1:a1(jj+1)-1);
% $$$             lfi = length(fi);
% $$$             i3=0;
% $$$             i4=0;
% $$$             %for i1 = lfi:-1:1
% $$$             for i1 = 1:lfi
% $$$                 cind1=indaz{fi(i1)};
% $$$                 Icind1=indIz{fi(i1)};
% $$$                 %indfi = lfi;
% $$$                 indfi = 1;
% $$$                 %for i2=length(Icind1):-1:1
% $$$                 for i2=1:length(Icind1)
% $$$                     if Icind1(i2)==jj  % Find the indeces for the jj'th rows in fi columns
% $$$                         i4=i4+1;
% $$$                         cindit(i4)=cind1(i2);
% $$$                         %break
% $$$                     end
% $$$                     %if indfi>0 && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
% $$$                     if indfi <= lfi && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
% $$$                         i3=i3+1;
% $$$                         cindt(i3)=cind1(i2);
% $$$                         %indfi = indfi-1;
% $$$                         indfi = indfi+1;
% $$$                     end
% $$$                 end
% $$$             end
% $$$             % remove extras
% $$$             cind=cindt(1:i3);
% $$$             cindi=cindit(1:i4);
% $$$ 
% $$$ 
% $$$             Zt = z(cind);
% $$$             Zt=reshape(Zt,lfi,lfi);
% $$$             zij = -fil'*Zt;
% $$$             z(cindi) = zij;
% $$$             indz = cumsum(histc(indIz{jj},[0 ; fi]));
% $$$             indz = az1(jj) + indz(1:end-1);
% $$$             z(indz) = zij;
% $$$ 
% $$$             zij = 1./ld(a1(jj)) - fil'*z(indz);
% $$$             z(az1(jj)-1+find(indIz{jj}==jj,1)) = zij;
% $$$         end
% $$$         
% $$$         Z = sparse(Iz,Jz,z);
% $$$         r(q) = 1:n;
% $$$         Z = Z(r,r);
% $$$         
% $$$ 
% $$$     end



        
        
        
        
        
        
% $$$         [LD, p, q] = ldlchol(A);
% $$$         [I,J,ld] = find(LD);
% $$$         temp = [I(:) J(:) ; J(:) I(:)];
% $$$         temp = sortrows(unique(temp,'rows'),2);
% $$$         Iz = temp(:,1); Jz = temp(:,2); 
% $$$         
% $$$         % Find the column starting points
% $$$         a1=zeros(n,1);
% $$$         a2 = cumsum(histc(J,1:n));
% $$$         a1(1) = 1; a1(2:end) = a2(1:end-1) + 1;
% $$$         az1=zeros(n,1);
% $$$         az2 = cumsum(histc(Jz,1:n));
% $$$         az1(1) = 1; az1(2:end) = az2(1:end-1) + 1;
% $$$         
% $$$         for j=1:n
% $$$             indaz{j} = az1(j):az2(j);
% $$$             indIz{j} = Iz(indaz{j})';
% $$$         end
% $$$ 
% $$$         % Evaluate the sparse inverse
% $$$         z = zeros(size(Iz));
% $$$         z(end) = 1./ld(end);
% $$$         for jj = n-1:-1:1
% $$$             fil = ld(a1(jj)+1:a1(jj+1)-1);
% $$$             fi = I(a1(jj)+1:a1(jj+1)-1);
% $$$             lfi = length(fi);
% $$$             indz = cumsum(histc(indIz{jj},[0 ; fi]));
% $$$             indz = az1(jj) + indz(1:end-1);
% $$$             for i1 = 1:lfi
% $$$                 cind1=indaz{fi(i1)};
% $$$                 Icind1=indIz{fi(i1)};
% $$$                 indfi = 1;
% $$$                 for i2=1:length(Icind1)
% $$$                     if Icind1(i2)==jj  % Find the indeces for the jj'th row in fi columns
% $$$                         cindit = cind1(i2);
% $$$                     end
% $$$                     if indfi <= lfi && fi(indfi) == Icind1(i2) % Find the indeces for the fi'th rows in i2'nd columns
% $$$                         zij = -fil(indfi).*z(cind1(i2));
% $$$                         z(cindit) = zij;
% $$$                         z(indz(indfi)) = zij; 
% $$$                         indfi = indfi+1;
% $$$                     end
% $$$                 end
% $$$             end
% $$$             zij = 1./ld(a1(jj)) - fil'*z(indz);
% $$$             z(az1(jj)-1+find(indIz{jj}==jj,1)) = zij;
% $$$         end
% $$$         
% $$$         Z = sparse(Iz,Jz,z);
% $$$         r(q) = 1:n;
% $$$         Z = Z(r,r);               
% $$$ 
% $$$ 
% $$$     end
