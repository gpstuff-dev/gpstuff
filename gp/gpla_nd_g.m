function [g, gdata, gprior] = gpla_nd_g(w, gp, x, y, varargin)
%GPLA_ND_G   Evaluate gradient of Laplace approximation's marginal
%            log posterior estimate for non-diagonal likelihoods (GPLA_ND_E)
%
%  Description
%    G = GPLA_ND_G(W, GP, X, Y, OPTIONS) takes a full GP
%    hyper-parameter vector W, structure GP a matrix X of
%    input vectors and a matrix Y of target vectors, and evaluates
%    the gradient G of EP's marginal log posterior estimate . Each
%    row of X corresponds to one input vector and each row of Y
%    corresponds to one target vector.
%
%    [G, GDATA, GPRIOR] = GPLA_ND_G(W, GP, X, Y, OPTIONS) also
%    returns the data and prior contributions to the gradient.
%
%    OPTIONS is optional parameter-value pair
%      z - optional observed quantity in triplet (x_i,y_i,z_i)
%          Some likelihoods may use this. For example, in case of
%          Poisson likelihood we have z_i=E_i, that is, expected
%          value for ith case.
%
%  See also
%    GPLA_ND_E, GPLA_E, GPLA_ND_PRED

% Copyright (c) 2010 Jaakko Riihim�ki, Pasi Jyl�nki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPLA_ND_G';
ip.addRequired('w', @(x) isvector(x) && isreal(x) && all(isfinite(x)));
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.parse(w, gp, x, y, varargin{:});
z=ip.Results.z;

gp = gp_unpak(gp, w);       % unpak the parameters
ncf = length(gp.cf);
n=size(x,1);
nout=size(y,2);

g = [];
gdata = [];
gprior = [];

% First Evaluate the data contribution to the error
switch gp.type
  % ============================================================
  % FULL
  % ============================================================
  case 'FULL'   % A full GP
    
    if isfield(gp, 'comp_cf')  % own covariance for each ouput component
      multicf = true;
      if length(gp.comp_cf) ~= nout && nout > 1
        error('GPLA_ND_G: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
      end
    else
      multicf = false;
    end
    
    if isfield(gp.lik, 'structW') && ~gp.lik.structW
      
      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        nl=n;
      else
        if isfield(gp.lik,'xtime')
          xtime=gp.lik.xtime;
          ntime = size(xtime,1);
          nl=[ntime n];
        else
          nl=[n n];
        end
      end
      nlp=length(nl); % number of latent processes
      
      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
          gptmp=gp; gptmp.jitterSigma2=0;
          Ka = gp_trcov(gptmp, unique(x(:,1)));
          wtmp=gp_pak(gptmp); wtmp(1)=0; gptmp=gp_unpak(gptmp,wtmp);
          Kb = gp_trcov(gptmp, unique(x(:,2)));
          clear gptmp
          n1=size(Ka,1);
          n2=size(Kb,1);
        else
          K = gp_trcov(gp,x);
        end
      else
        K = zeros(sum(nl));
        if isfield(gp.lik,'xtime')
          K(1:ntime,1:ntime)=gp_trcov(gp, xtime, gp.comp_cf{1});
          K((1:n)+ntime,(1:n)+ntime) = gp_trcov(gp, x, gp.comp_cf{2});
        else
          for i1=1:nlp
            K((1:n)+(i1-1)*n,(1:n)+(i1-1)*n) = gp_trcov(gp, x, gp.comp_cf{i1});
          end
        end
      end
      
      if isfield(gp,'meanf')
        [H,b_m,B_m]=mean_prep(gp,x,[]);
        Hb_m=H'*b_m;
        if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
          % only zero mean function implemented for Kronecker
          % approximation
          iKHb_m=zeros(n,1);
        else
          K=K+H'*B_m*H;
          iKHb_m=K\Hb_m;
        end
      end
      
      [e, edata, eprior, f, L, a, W, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
      if isnan(e)
        return
      end
      
      if isfield(gp.lik, 'fullW') && gp.lik.fullW
        g2 = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
        ny=sum(y);
        
        g3=feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z);
        
        if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
          
          [Va,Da]=eig(Ka); [Vb,Db]=eig(Kb);
          % eigenvalues of K matrix
          Dtmp=kron(diag(Da),diag(Db));
          [sDtmp,istmp]=sort(Dtmp,'descend');
          
          % Form the low-rank approximation. Exclude eigenvalues
          % smaller than gp.latent_opt.eig_tol or take
          % gp.latent_opt.eig_prct*n eigenvalues at most.
          nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
          sDtmp=sDtmp+gp.jitterSigma2;
          
          itmp1=meshgrid(1:n1,1:n2);
          itmp2=meshgrid(1:n2,1:n1)';
          ind=[itmp1(:) itmp2(:)];
          
          % included eigenvalues
          Dlr=sDtmp(1:nlr);
          % included eigenvectors
          Vlr=zeros(n,nlr);
          for i1=1:nlr
            Vlr(:,i1)=kron(Va(:,ind(istmp(i1),1)),Vb(:,ind(istmp(i1),2)));
          end
          
          Lb=gp_trvar(gp,x)-sum(bsxfun(@times,Vlr.*Vlr,Dlr'),2);
          
          if isfield(gp,'meanf')
            Dt=[Dlr; diag(B_m)];
            Vt=[Vlr H'];
          else
            Dt=Dlr;
            Vt=Vlr;
          end
          
          Lbt=ny*(g2)+1./Lb;
          
          St=[diag(1./Dt)+Vt'*bsxfun(@times,1./Lb,Vt) zeros(size(Dt,1),1); ...
            zeros(1,size(Dt,1)) 1];
          Pt=[bsxfun(@times,1./Lb,Vt) sqrt(ny)*g2];
          Ptt=bsxfun(@times,1./sqrt(Lbt),Pt);
          
          StL=chol(St-Ptt'*Ptt,'lower');
          iStL=StL\(bsxfun(@times,Pt',1./Lbt'));
          
          dA=(1./Lbt+sum(iStL.*iStL)');
          iStLg3=iStL*g3;
          const1=( 0.5*ny*(sum( dA.*g3))-ny*(g3'*(g3./Lbt)+iStLg3'*iStLg3) );
          const2=(g3./Lbt)+iStL'*iStLg3;
         
          s2=const1.*g3 - 0.5*ny*dA.*g3 + ny*const2.*g3;
          
        else
          
          KW=-(K*(sqrt(ny)*g2))*(sqrt(ny)*g2)'- bsxfun(@times, K, (-ny*g2)');
          KW(1:(n+1):end)=KW(1:(n+1):end)+1;
          iKW=KW\eye(n);
          A=iKW*K;
          
          const1=( 0.5*ny*(sum(A(1:(n+1):end).*g3'))-ny*sum(sum(A.*(g3*g3'))) );
          const2=sum(bsxfun(@times,A,g3));
          s2=const1.*g3 - 0.5*ny*diag(A).*g3 + ny*const2'.*g3;
        end
        
      else
        KW=zeros(sum(nl));
        if isfield(gp.lik,'xtime')
          [Wdiag, Wmat] = feval(gp.lik.fh.llg2, gp.lik, y, f, 'latent', z);
          Wdiag=-Wdiag; Wmat=-Wmat;
          KW=zeros(ntime+n);
          KW(1:ntime,1:ntime)=bsxfun(@times, K(1:ntime,1:ntime), Wdiag(1:ntime)');
          KW(1:ntime,ntime+(1:n))=K(1:ntime,1:ntime)*Wmat;
          KW(ntime+(1:n),1:ntime)=K(ntime+(1:n),ntime+(1:n))*Wmat';
          KW(ntime+(1:n),ntime+(1:n))=bsxfun(@times,K(ntime+(1:n),ntime+(1:n)), Wdiag(ntime+(1:n))');
          KW(1:(ntime+n+1):end)=KW(1:(ntime+n+1):end)+1;
        else
          for il=1:nlp
            KW((1:n)+(il-1)*n,1:n)=bsxfun(@times, K((1:n)+(il-1)*n,(1:n)+(il-1)*n), W(1:n,il)');
            KW((1:n)+(il-1)*n,(n+1):(2*n))=bsxfun(@times, K((1:n)+(il-1)*n,(1:n)+(il-1)*n), W((n+1):(2*n),il)');
          end
          KW(1:(2*n+1):end)=KW(1:(2*n+1):end)+1;
        end
        
        iKW=KW\eye(sum(nl));
        A=iKW*K;
        s2=zeros(sum(nl),1);
        
        if isfield(gp.lik,'xtime')
          A_diag=diag(A);
          A_mat=A(1:ntime,ntime+(1:n));
          for i1=1:sum(nl)
            [dw_diag,dw_mat]=feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z, i1);
            s2(i1) = 0.5*(sum(A_diag.*dw_diag)+2*sum(sum(A_mat.*dw_mat)));
          end
        else
          dw_mat = feval(gp.lik.fh.llg3, gp.lik, y, f, 'latent', z);
          for i1=1:n
            s2(i1) = 0.5*trace(A(i1:n:(i1+n),i1:n:(i1+n))*dw_mat(:,:,1,i1));
            s2(i1+n) = 0.5*trace(A(i1:n:(i1+n),i1:n:(i1+n))*dw_mat(:,:,2,i1));
          end
        end
      end
      
      % =================================================================
      % Gradient with respect to covariance function parameters
      if ~isempty(strfind(gp.infer_params, 'covariance'))
        % Evaluate the gradients from covariance functions
        for i=1:ncf
          
          i1=0;
          if ~isempty(gprior)
            i1 = length(gprior);
          end
          
          gpcf = gp.cf{i};
          if isfield(gp.lik, 'fullW') && gp.lik.fullW
            if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
              
                gptmp=gp; gptmp.jitterSigma2=0;
                DKa = feval(gpcf.fh.cfg, gptmp.cf{1}, unique(x(:,1)));
                wtmp=gp_pak(gptmp); wtmp(1)=0; gptmp=gp_unpak(gptmp,wtmp);
                DKb = feval(gpcf.fh.cfg, gptmp.cf{1}, unique(x(:,2)));
                clear gptmp
                
                for j1=1:length(DKa)
                  [DVa{j1},DDa{j1}]=eig(DKa{j1});
                end
                for j1=1:length(DKb)
                  [DVb{j1},DDb{j1}]=eig(DKb{j1});
                end
                
                % low-rank approximation of the derivative matrix w.r.t.
                % magnitude hyperparameter, low-rank + diagonal correction
                Datmp=kron(diag(DDa{1}),diag(Db));
                [sDatmp,isatmp]=sort(Datmp,'descend');
                nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
                Dmslr=sDatmp(1:nlr);
                Vmslr=zeros(n,nlr);
                for j1=1:nlr
                  Vmslr(:,j1)=kron(DVa{1}(:,ind(isatmp(j1),1)),Vb(:,ind(isatmp(j1),2)));
                end
                % diagonal correction
                dc=feval(gpcf.fh.cfg, gpcf, x(1,:));
                Lms=dc{1}*ones(n,1)-sum(bsxfun(@times,Vmslr.^2,Dmslr'),2);
                
                
                % low-rank approximation of the derivative matrix w.r.t.
                % lengthscale hyperparameter, low-rank1 + lowr-rank2 + diagonal correction
                %
                % first low-rank part
                Datmp=kron(diag(DDa{2}),diag(Db));
                [sDatmp,isatmp]=sort(abs(Datmp),'descend');
                nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
                sDatmp=Datmp(isatmp);
                Dlslr1=sDatmp(1:nlr);
                Vlslr1=zeros(n,nlr);
                for j1=1:nlr
                  Vlslr1(:,j1)=kron(DVa{2}(:,ind(isatmp(j1),1)),Vb(:,ind(isatmp(j1),2)));
                end
                % second low-rank part
                Dbtmp=kron(diag(Da),diag(DDb{2}));
                [sDbtmp,isbtmp]=sort(abs(Dbtmp),'descend');
                nlr=min([sum(sDtmp>gp.latent_opt.eig_tol) round(gp.latent_opt.eig_prct*n)]);
                sDbtmp=Dbtmp(isbtmp);
                Dlslr2=sDbtmp(1:nlr);
                Vlslr2=zeros(n,nlr);
                for j1=1:nlr
                  Vlslr2(:,j1)=kron(Va(:,ind(isbtmp(j1),1)),DVb{2}(:,ind(isbtmp(j1),2)));
                end
                % diagonal correction
                Lls=dc{2}*ones(n,1)-sum(bsxfun(@times,Vlslr1.^2,Dlslr1'),2)-sum(bsxfun(@times,Vlslr2.^2,Dlslr2'),2);
                
            else
              DKff = feval(gpcf.fh.cfg, gpcf, x);
            end
          else
            % check in which components the covariance function is present
            do = false(nlp,1);
            for z1=1:nlp
              if any(gp.comp_cf{z1}==i)
                do(z1) = true;
              end
            end
            
            if isfield(gp.lik,'xtime')
              if ~isempty(intersect(gp.comp_cf{1},i))
                DKff = feval(gpcf.fh.cfg, gpcf, xtime);
              else
                DKff = feval(gpcf.fh.cfg, gpcf, x);
              end
            else
              DKff = feval(gpcf.fh.cfg, gpcf, x);
            end
          end
          gprior_cf = -feval(gpcf.fh.lpg, gpcf);
          g1 = feval(gp.lik.fh.llg, gp.lik, y, f, 'latent', z);
          
          
          if isfield(gp.lik, 'fullW') && gp.lik.fullW
            
            if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
              
              for i2 = 1:length(DDa)
                i1 = i1+1;
                
                if ~isfield(gp,'meanf')
                  if i2==1
                    % derivative wrt magnitude hyperparameter
                    Vmsa = Vmslr'*a;
                    s1a = 0.5*(Vmsa'*(bsxfun(@times,Dmslr,Vmsa)) + a'*(Lms.*a));
                  elseif i2==2
                    % derivative wrt lengthscale hyperparameter
                    Vls1 = Vlslr1'*a;
                    Vls2 = Vlslr2'*a;
                    s1a = 0.5*( Vls1'*bsxfun(@times,Dlslr1,Vls1) + Vls2'*bsxfun(@times,Dlslr2,Vls2) + a'*(Lls.*a));
                  end
                else
                  if i2==1
                    % derivative wrt magnitude hyperparameter
                    Vmsa = Vmslr'*(a-iKHb_m);
                    s1a = 0.5*(Vmsa'*(bsxfun(@times,Dmslr,Vmsa)) + (a-iKHb_m)'*(Lms.*(a-iKHb_m)));
                  elseif i2==2
                    % derivative wrt lengthscale hyperparameter
                    Vls1 = Vlslr1'*(a-iKHb_m);
                    Vls2 = Vlslr2'*(a-iKHb_m);
                    s1a = 0.5*( Vls1'*bsxfun(@times,Dlslr1,Vls1) + Vls2'*bsxfun(@times,Dlslr2,Vls2) + (a-iKHb_m)'*(Lls.*(a-iKHb_m)));
                  end
                end
                
                % DKg2=DKff{i2}*g2;
                if i2==1
                  DKg2 = Vmslr*bsxfun(@times,Dmslr,Vmslr'*g2) + Lms.*g2;
                elseif i2==2
                  DKg2 =  Vlslr1*bsxfun(@times,Dlslr1,Vlslr1'*g2) + Vlslr2*bsxfun(@times,Dlslr2,Vlslr2'*g2) + Lls.*g2;
                end
                
                WDKg2=ny*(g2.*DKg2-(g2*(g2'*DKg2)));
                s1b = -0.5*(ny)*( ( - (DKg2-((WDKg2./Lbt)+(iStL'*(iStL*WDKg2)))))'*(g2) );
                
                if i2==1 % magnitude hyperparameter
                  
                  % low-rank
                  WDVa=ny*( bsxfun(@times,g2,Vmslr)-g2*(g2'*Vmslr) );
                  stmp=Vmslr-(bsxfun(@times,(1./Lbt),WDVa)+(iStL'*(iStL*WDVa)));
                  s1clr = 0.5*sum( (sum(bsxfun(@times,stmp,Dmslr').*Vmslr,2))' .*(-ny*g2)' );
                  
                  % diagonal correction
                  s1cdtmp = Lms - ( ny*( (g2.*Lms)./Lbt  - (g2./Lbt).*(g2'.*Lms')' ) + ...
                    sum(iStL.* (ny*( bsxfun(@times,iStL,(g2.*Lms)') - (iStL*g2)*(g2'.*Lms') )) )' );
                  s1cd=0.5*sum( s1cdtmp' .*(-ny*g2)' );
                  s1c=s1clr+s1cd;
                  DKg = Vmslr*bsxfun(@times,Dmslr,Vmslr'*g1) + Lms.*g1;
                  
                  
                elseif i2==2 % lengthscale hyperparameter
                  
                  % low-rank 1
                  WDVa=ny*( bsxfun(@times,g2,Vlslr1)-g2*(g2'*Vlslr1) );
                  stmp=Vlslr1-(bsxfun(@times,(1./Lbt),WDVa)+(iStL'*(iStL*WDVa)));
                  s1clr1 = 0.5*sum( (sum(bsxfun(@times,stmp,Dlslr1').*Vlslr1,2))' .*(-ny*g2)' );
                  
                  % low-rank 2
                  WDVb=ny*( bsxfun(@times,g2,Vlslr2)-g2*(g2'*Vlslr2) );
                  stmp=Vlslr2-(bsxfun(@times,(1./Lbt),WDVb)+(iStL'*(iStL*WDVb)));
                  s1clr2 = 0.5*sum( (sum(bsxfun(@times,stmp,Dlslr2').*Vlslr2,2))' .*(-ny*g2)' );
                  
                  % diagonal correction
                  s1cdtmp = Lls - ( ny*( (g2.*Lls)./Lbt  - (g2./Lbt).*(g2'.*Lls')' ) + ...
                    sum(iStL.* (ny*( bsxfun(@times,iStL,(g2.*Lls)') - (iStL*g2)*(g2'.*Lls') )) )' );
                  s1cd=0.5*sum( s1cdtmp' .*(-ny*g2)' );
                  
                  s1c=s1clr1+s1clr2+s1cd;
                  
                  DKg = Vlslr1*bsxfun(@times,Dlslr1,Vlslr1'*g1) + Vlslr2*bsxfun(@times,Dlslr2,Vlslr2'*g1) + Lls.*g1;
                  
                end
                
                s1=s1a+s1b+s1c;
                
                %DKg=DKff{i2}*g1;
                WDKg=ny*(g2.*DKg-(g2*(g2'*DKg)));
                s3=DKg-((WDKg./Lbt)+(iStL'*(iStL*WDKg)));
                
                gdata(i1) = -(s1 + s2'*s3);
                gprior(i1) = gprior_cf(i2);
              end
              
            else
              
              for i2 = 1:length(DKff)
                i1 = i1+1;
                if ~isfield(gp,'meanf')
                  s1 = 0.5 * a'*DKff{i2}*a - 0.5*((-iKW*(DKff{i2}*(sqrt(ny)*g2)))'*(sqrt(ny)*g2)) + 0.5*sum(sum(iKW'.*DKff{i2}).*(-ny*g2)');
                else
                  s1 = 0.5 * (a-iKHb_m)'*DKff{i2}*(a-iKHb_m) - 0.5*((-iKW*(DKff{i2}*(sqrt(ny)*g2)))'*(sqrt(ny)*g2)) + 0.5*sum(sum(iKW'.*DKff{i2}).*(-ny*g2)');
                end
                %b = DKff{i2} * g1;
                if issparse(K)
                  s3 = b - K*(sqrtW*ldlsolve(L,sqrtW*b));
                else
                  s3=iKW*(DKff{i2}*g1);
                end
              
              gdata(i1) = -(s1 + s2'*s3);
              gprior(i1) = gprior_cf(i2);
              end
            end
          else
            if isfield(gp.lik,'xtime')
              WiKW=zeros(sum(nl));
              WiKW(1:ntime,1:ntime)=bsxfun(@times, Wdiag(1:ntime),iKW(1:ntime,1:ntime)) + Wmat*iKW(ntime+(1:n),1:ntime);
              WiKW(1:ntime,ntime+(1:n))=bsxfun(@times, Wdiag(1:ntime),iKW(1:ntime,ntime+(1:n))) + Wmat*iKW(ntime+(1:n),ntime+(1:n));
              WiKW(ntime+(1:n),1:ntime)=Wmat'*iKW(1:ntime,1:ntime) + bsxfun(@times, Wdiag(ntime+(1:n)),iKW(ntime+(1:n),1:ntime));
              WiKW(ntime+(1:n),ntime+(1:n))=bsxfun(@times, Wdiag(ntime+(1:n)),iKW(ntime+(1:n),ntime+(1:n))) + Wmat'*iKW(1:ntime,ntime+(1:n));
            else
              WiKW11=bsxfun(@times,W(1:n,1),iKW(1:n,1:n))+bsxfun(@times,W(1:n,2),iKW((1:n)+n,1:n));
              WiKW12=bsxfun(@times,W(1:n,1),iKW(1:n,(1:n)+n))+bsxfun(@times,W(1:n,2),iKW((1:n)+n,(1:n)+n));
              WiKW21=bsxfun(@times,W((1:n)+n,1),iKW(1:n,1:n))+bsxfun(@times,W((1:n)+n,2),iKW((1:n)+n,1:n));
              WiKW22=bsxfun(@times,W((1:n)+n,1),iKW(1:n,(1:n)+n))+bsxfun(@times,W((1:n)+n,2),iKW((1:n)+n,(1:n)+n));
              WiKW=[WiKW11 WiKW12; WiKW21 WiKW22];
            end
            
            for i2 = 1:length(DKff)
              i1 = i1+1;
              if ~isfield(gp,'meanf')
                dKnl = zeros(sum(nl));
                if isfield(gp.lik,'xtime')
                  if ~isempty(intersect(gp.comp_cf{1},i)) %do(indnl)
                    dKnl(1:ntime,1:ntime) = DKff{i2};
                    %end
                  else
                    %if do(indnl)
                    dKnl(ntime+(1:n),ntime+(1:n)) = DKff{i2};
                    %end
                  end
                else
                  for indnl=1:nlp
                    if do(indnl)
                      dKnl((1:n)+(indnl-1)*n,(1:n)+(indnl-1)*n) = DKff{i2};
                    end
                  end
                end
                s1 = 0.5 * a'*dKnl*a - 0.5*sum(sum((dKnl.*WiKW)));
              else
                %s1 = 0.5 * (a-K\(H'*b_m))'*DKff{i2}*(a-K\(H'*b_m)) - 0.5*sum(sum(R.*DKff{i2}));
              end
              b = dKnl*g1;
              %b = DKff{i2} * g1;
              if issparse(K)
                s3 = b - K*(sqrtW*ldlsolve(L,sqrtW*b));
              else
                s3=iKW*b;
              end
              
              gdata(i1) = -(s1 + s2'*s3);
              gprior(i1) = gprior_cf(i2);
            end
          end
          
          if isfield(gp.latent_opt, 'kron') && gp.latent_opt.kron==1
            % Set the gradients of hyperparameter
            if length(gprior_cf) > length(DKa)
              for i2=length(DKff)+1:length(gprior_cf)
                i1 = i1+1;
                gdata(i1) = 0;
                gprior(i1) = gprior_cf(i2);
              end
            end
          else
            % Set the gradients of hyperparameter
            if length(gprior_cf) > length(DKff)
              for i2=length(DKff)+1:length(gprior_cf)
                i1 = i1+1;
                gdata(i1) = 0;
                gprior(i1) = gprior_cf(i2);
              end
            end
          end
        end
      end
      
      % =================================================================
      % Gradient with respect to likelihood function parameters
      if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
          && ~isempty(gp.lik.fh.pak(gp.lik))
        
        gdata_lik = 0;
        lik = gp.lik;
        
        g_logPrior = -feval(lik.fh.lpg, lik);
        if ~isempty(g_logPrior)
          
          DW_sigma = feval(lik.fh.llg3, lik, y, f, 'latent2+param', z);
          DL_sigma = feval(lik.fh.llg, lik, y, f, 'param', z);
          b = K * feval(lik.fh.llg2, lik, y, f, 'latent+param', z);
          s3 = iKW*b;
          
          gdata_lik = - DL_sigma - 0.5.*sum(sum((A.*DW_sigma))) - s2'*s3;
          
          % set the gradients into vectors that will be returned
          gdata = [gdata gdata_lik];
          gprior = [gprior g_logPrior];
          i1 = length(g_logPrior);
          i2 = length(gdata_lik);
          if i1  > i2
            gdata = [gdata zeros(1,i1-i2)];
          end
        end
      end
      
      % g = gdata + gprior;
      
    else
      
      K = zeros(n,n,nout);
      if multicf
        for i1=1:nout
          K(:,:,i1) = gp_trcov(gp, x, gp.comp_cf{i1});
        end
      else
        Ktmp=gp_trcov(gp, x);
        for i1=1:nout
          K(:,:,i1) = Ktmp;
        end
      end
      
      [e, edata, eprior, f, L, a, E, M, p] = gpla_nd_e(gp_pak(gp), gp, x, y, 'z', z);
      
      % softmax
      f2=reshape(f,n,nout);
      
      llg = gp.lik.fh.llg(gp.lik, y, f2, 'latent', z);
      [pi2_vec, pi2_mat] = gp.lik.fh.llg2(gp.lik, y, f2, 'latent', z);
      R = repmat(1./pi2_vec,1,n).*pi2_mat;
      RE = zeros(n,n*nout);
      for i1=1:nout
        RE(:,(1:n)+(i1-1)*n) = R((1:n)+(i1-1)*n,:)'*E(:,:,i1);
      end
      
      inv_iWK=zeros(n,n,nout);
      
      % Matrices for computing the derivative of determinant term w.r.t. f
      A=zeros(nout, nout, n);
      Minv=M\(M'\eye(n));
      Minv=(Minv+Minv')./2;
      for cc1=1:nout
        EMinv=RE(:,(1:n)+(cc1-1)*n)'*Minv;
        KEMinv=K(:,:,cc1)*EMinv;
        for cc2=1:nout
          if cc2>=cc1
            if cc1==cc2
              EMtmp = - EMinv*RE(:,(1:n)+(cc2-1)*n);
              EMtmp = EMtmp + E(:,:,cc1);
              inv_iWK(:,:,cc1) = EMtmp;
              A(cc1,cc1,:) = diag(K(:,:,cc1))-sum((K(:,:,cc1)*EMtmp).*K(:,:,cc1),2);
            else
              EMtmp = - KEMinv*RE(:,(1:n)+(cc2-1)*n);
              A(cc1,cc2,:) = -sum(EMtmp.*K(:,:,cc2),2);
              A(cc2,cc1,:) = -sum(EMtmp.*K(:,:,cc2),2);
            end
          end
        end
      end
      
      % Derivative of determinant term w.r.t. f
      s2=zeros(n*nout,1);
      dw_mat = gp.lik.fh.llg3(gp.lik, y, f, 'latent', z);
      for cc3=1:nout
        for ii1=1:n
          s2(ii1+(cc3-1)*n) = -0.5*trace(A(:,:,ii1)*dw_mat(:,:,cc3,ii1));
        end
      end
      
      % Loop over the covariance functions
      for i=1:ncf
        DKllg=zeros(size(a));
        EDKllg=zeros(size(a));
        DKffba=zeros(n*nout,1);
        
        % check in which components the covariance function is present
        do = false(nout,1);
        if multicf
          for z1=1:nout
            if any(gp.comp_cf{z1}==i)
              do(z1) = true;
            end
          end
        else
          do = true(nout,1);
        end
        
        i1=0;
        if ~isempty(gprior)
          i1 = length(gprior);
        end
        
        % Gradients from covariance functions
        gpcf = gp.cf{i};
        DKff = gpcf.fh.cfg(gpcf, x);
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:length(DKff)
          i1 = i1+1;
          DKffb=DKff{i2};
          
          % Derivative of explicit terms
          trace_sum_tmp=0;
          for z1=1:nout
            if do(z1)
              DKffba((1:n)+(z1-1)*n)=DKffb*a((1:n)+(z1-1)*n);
              trace_sum_tmp = trace_sum_tmp + sum(sum( inv_iWK(:,:,z1) .* DKffb ));
            end
          end
          s1 = 0.5 * a'*DKffba - 0.5.*trace_sum_tmp;
          
          % Derivative of f w.r.t. theta
          for z1=1:nout
            if do(z1)
              DKllg((1:n)+(z1-1)*n)=DKffb*llg((1:n)+(z1-1)*n);
              EDKllg((1:n)+(z1-1)*n)=E(:,:,z1)*DKllg((1:n)+(z1-1)*n);
            end
          end
          s3 = EDKllg - RE'*(M\(M'\(RE*DKllg)));
          for z1=1:nout
            s3((1:n)+(z1-1)*n)=K(:,:,z1)*s3((1:n)+(z1-1)*n);
          end
          s3 = DKllg - s3;
          
          gdata(i1) = -(s1 + s2'*s3);
          gprior(i1) = gprior_cf(i2);
          
        end
        
        % Set the gradients of hyper-hyperparameter
        if length(gprior_cf) > length(DKff)
          for i2=length(DKff)+1:length(gprior_cf)
            i1 = i1+1;
            gdata(i1) = 0;
            gprior(i1) = gprior_cf(i2);
          end
        end
      end
      
    end
    
    %         % =================================================================
    %         % Gradient with respect to likelihood function parameters
    %         if ~isempty(strfind(gp.infer_params, 'likelihood')) ...
    %             && ~isempty(gp.lik.fh.pak(gp.lik))
    %
    %             gdata_likelih = 0;
    %             lik = gp.lik;
    %
    %             g_logPrior = feval(lik.fh.gprior, lik);
    %             if ~isempty(g_logPrior)
    %
    %                 DW_sigma = feval(lik.fh.llg3, lik, y, f, 'latent2+hyper', z);
    %                 DL_sigma = feval(lik.fh.llg, lik, y, f, 'hyper', z);
    %                 b = K * feval(lik.fh.llg2, lik, y, f, 'latent+hyper', z);
    %                 s3 = b - K*(R*b);
    %                 nl= size(DW_sigma,2);
    %
    %                 gdata_lik = - DL_sigma - 0.5.*sum(repmat(C2,1,nl).*DW_sigma) - s2'*s3;
    %
    %                 % set the gradients into vectors that will be returned
    %                 gdata = [gdata gdata_lik];
    %                 gprior = [gprior g_logPrior];
    %                 i1 = length(g_logPrior);
    %                 i2 = length(gdata_lik);
    %                 if i1  > i2
    %                     gdata = [gdata zeros(1,i1-i2)];
    %                 end
    %             end
    %         end
    
    g = gdata + gprior;
    
    % ============================================================
    % FIC
    % ============================================================
  case 'FIC'
    
    % ============================================================
    % PIC
    % ============================================================
  case {'PIC' 'PIC_BLOCK'}
    
    % ============================================================
    % CS+FIC
    % ============================================================
  case 'CS+FIC'
    
end

assert(isreal(gdata))
assert(isreal(gprior))
end
