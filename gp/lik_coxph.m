function lik = lik_coxph(varargin)
%LIK_COXPH    Create a Cox proportional hazard likelihood structure
%
%  Description
%    LIK = LIK_COXPH('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a proportional hazard model where a piecewise log-constant
%    baseline hazard is assumed.  
%    
%    The likelihood contribution for the ith observation is
%
%      l_i = h_i(y_i)^(1-z_i)*[exp(-int_0^y_i*h_i dt],
%    
%    where hazard is h_i=h_0(y_i)*exp(f_i). A zero mean Gaussian process
%    prior is placed for f = [f_1, f_2,...,f_n] ~ N(0, C). C is the
%    covariance matrix, whose elements are given as C_ij = c(x_i, x_j |
%    th). The function c(x_i, x_j| th) is covariance function and th its
%    parameters, hyperparameters. We place a hyperprior for
%    hyperparameters, p(th). 
%
%    The time axis is partioned into K intervals with equal lengths:
%    0 = s_0 < s_1 < ... < s_K, where s_K > y_i for all i. The baseline
%    hazard rate function h_0 is piecewise constant,  
%
%      h_0(t) = la_k,
%
%    when t belongs to the interval (s_{k-1},s_k] and where ft_k=log(la_k).
%    The hazard rate function is smoothed by assuming another Gaussian
%    process prior ft = [ft_1, ft_2,...,ft_K] ~ N(0, C). 
%
%    z is a vector of censoring indicators with z = 0 for uncensored event
%    and z = 1 for right censored event. 
%
%    When using the Coxph likelihood you need to give the vector z
%    as an extra parameter to each function that requires also y. 
%    For example, you should call gpla_e as follows: gpla_e(w, gp,
%    x, y, 'z', z)
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%

% Copyright (c) 2007-2010 Jarno Vanhatalo & Jouni Hartikainen
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2011 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_COXPH';
  ip.addOptional('lik', [], @isstruct);
  ip.parse(varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'Coxph';
    lik.structW = false;
  else
    if ~isfield(lik,'type') && ~isequal(lik.type,'Coxph')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  if init
    % Set the function handles to the nested functions
    lik.fh.pak = @lik_coxph_pak;
    lik.fh.unpak = @lik_coxph_unpak;
%    lik.fh.lp = @lik_coxph_lp;
%    lik.fh.lpg = @lik_coxph_lpg;
    lik.fh.ll = @lik_coxph_ll;
    lik.fh.llsamps = @lik_coxph_llsamps;
    lik.fh.llg = @lik_coxph_llg;    
    lik.fh.llg2 = @lik_coxph_llg2;
    lik.fh.llg3 = @lik_coxph_llg3;
    lik.fh.tiltedMoments = @lik_coxph_tiltedMoments;
    lik.fh.predy = @lik_coxph_predy;
    lik.fh.invlink = @lik_coxph_invlink;
    lik.fh.recappend = @lik_coxph_recappend;
  end

  function [w,s] = lik_coxph_pak(lik)
  %LIK_COXPH_PAK  Combine likelihood parameters into one vector.
  %
  %  Description 
  %    W = LIK_COXPH_PAK(LIK) takes a likelihood structure LIK and
  %    combines the parameters into a single row vector W.
  %     
  %       w = log(lik.disper)
  %
  %   See also
  %   LIK_COXPH_UNPAK, GP_PAK
    
    w=[];s={};
  end


  function [lik, w] = lik_coxph_unpak(lik, w)
  %LIK_COXPH_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    [LIK, W] = LIK_COXPH_UNPAK(W, LIK) takes a likelihood
  %    structure LIK and extracts the parameters from the vector W
  %    to the LIK structure.
  %     
  %   Assignment is inverse of  
  %       w = log(lik.disper)
  %
  %   See also
  %   LIK_COXPH_PAK, GP_UNPAK
  
    lik=lik;
    w=w;
    
  end
  
  function ll = lik_coxph_ll(lik, y, f, z)
  %LIK_COXPH_LL  Log likelihood
  %
  %  Description
  %    LL = LIK_COXPH_LL(LIK, Y, F, Z) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z, and
  %    latent values F. Returns the log likelihood, log p(y|f,z).
  %
  %  See also
  %    LIK_COXPH_LLG, LIK_COXPH_LLG3, LIK_COXPH_LLG2, GPLA_E
    
    if isempty(z)
      error(['lik_coxph -> lik_coxph_ll: missing z!    '... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input z. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end
    
    ntime=size(lik.xtime,1);
    
    [n,ny]=size(y);
    
    f1=f(1:ntime);
    f2=f((ntime+1):(ntime+n));
    
    la1=exp(f1);
    eta2=exp(f2);
    
    nu=1-z;
    sd=lik.stime(2)-lik.stime(1);
    
    if ny==1
      ll=0;
      for i1=1:n
        si=sum(y(i1)>lik.stime);
        ll=ll + nu(i1).*(f1(si)+f2(i1)) - (y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
      end
    else
      
      ll=0;
      sb=sum(bsxfun(@gt,y(:,1),lik.stime),2);
      se=sum(bsxfun(@gt,y(:,2),lik.stime),2);
      for i1=1:n
        %sb=sum(y(i1,1)>lik.stime); % begin
        %se=sum(y(i1,2)>lik.stime); % end
        %[i1 sb se se-sb]
        
        if sb(i1)==0
          ll=ll + nu(i1).*(f1(se(i1))+f2(i1)) - (y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1(1:(se(i1)-1)).*eta2(i1));
        else
          
          if se(i1)==sb(i1)
            ll=ll + nu(i1).*(f1(se(i1))+f2(i1)) - (y(i1,2)-y(i1,1)).*la1(se(i1)).*eta2(i1);
          else
            ll=ll + nu(i1).*(f1(se(i1))+f2(i1)) - (y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1((sb(i1)+1):(se(i1)-1)).*eta2(i1)) - (lik.stime(sb(i1)+1)-y(i1,1)).*la1(sb(i1)).*eta2(i1);
          end
        end
      end
    end
  end

  function ll = lik_coxph_llsamps(lik, y, f, z)
  %LIK_COXPH_LL  Log likelihood
  %
  %  Description
  %    LL = LIK_COXPH_LL(LIK, Y, F, Z) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z, and
  %    latent values F. Returns the log likelihood, log p(y|f,z).
  %
  %  See also
  %    LIK_COXPH_LLG, LIK_COXPH_LLG3, LIK_COXPH_LLG2, GPLA_E
    
    if isempty(z)
      error(['lik_coxph -> lik_coxph_ll: missing z!    '... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input z. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end
    
    ntime=size(lik.xtime,1);
    [n,ny]=size(y);
    
    i1=1;
    si=sum(y(i1)>lik.stime);
    
    f1=f(1:si);
    f2=f(end);
    
    la1=exp(f1);
    eta2=exp(f2);
    
    nu=1-z;
    sd=lik.stime(2)-lik.stime(1);

    
    ll=nu(i1).*(f1(si)+f2(i1)) - (y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
  end



  function llg = lik_coxph_llg(lik, y, f, param, z)
  %LIK_COXPH_LLG  Gradient of the log likelihood
  %
  %  Description 
  %    LLG = LIK_COXPH_LLG(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z and
  %    latent values F. Returns the gradient of the log likelihood
  %    with respect to PARAM. At the moment PARAM can be 'param' or
  %    'latent'.
  %
  %  See also
  %    LIK_COXPH_LL, LIK_COXPH_LLG2, LIK_COXPH_LLG3, GPLA_E

    if isempty(z)
      error(['lik_coxph -> lik_coxph_llg: missing z!    '... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input z. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end

    ntime=size(lik.xtime,1);
    
    [n,ny]=size(y);
    f1=f(1:ntime);
    f2=f((ntime+1):(ntime+n));
    
    la1=exp(f1);
    eta2=exp(f2);
    
    nu=1-z;
    sd=lik.stime(2)-lik.stime(1);
    llg=zeros(ntime+n,1);
    
    switch param
      case 'latent'
          
      if ny==1
        for i1=1:ntime
          ind=y>=lik.stime(i1) & y<lik.stime(i1+1);
          llg(i1)= sum(nu(ind) - (y(ind)-lik.stime(i1)).*la1(i1).*eta2(ind)) - sum(sum(sd.*la1(i1)).*eta2(~ind & y>=lik.stime(i1+1)));
        end
        for i1=1:n
          si=sum(y(i1)>lik.stime);
          llg(i1+ntime)= nu(i1) - (y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
        end
      else
        for i1=1:ntime
          
          % left truncated + follow-up entry: (1)
          ind_vkst = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1+1);
          % follow-up entry + follow-up exit: (2)
          ind_stsp = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
          % follow-up: (3)
          ind_s = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1+1);
          % follow-up exit: (4)
          ind_sp = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
          
          % (1)
          s2b=sum(-(lik.stime(i1+1)-y(ind_vkst,1)).*la1(i1).*eta2(ind_vkst));
          % (2)
          s3b=sum(nu(ind_stsp)) - sum((y(ind_stsp,2)-y(ind_stsp,1)).*la1(i1).*eta2(ind_stsp));
          % (3)
          s4= - sum(sd.*la1(i1).*eta2(ind_s));
          % (4)
          s5=sum(nu(ind_sp)) - sum((y(ind_sp,2)-lik.stime(i1)).*la1(i1).*eta2(ind_sp));
          
          llg(i1) = s2b+s3b+s4+s5;
        end
        
        sb=sum(bsxfun(@gt,y(:,1),lik.stime),2);
        se=sum(bsxfun(@gt,y(:,2),lik.stime),2);
        for i1=1:n
          %sb=sum(y(i1,1)>lik.stime); % begin
          %se=sum(y(i1,2)>lik.stime); % end
          if sb(i1)==0
            llg(i1+ntime)= nu(i1) - (y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1(1:(se(i1)-1)).*eta2(i1));
          else
            if se(i1)==sb(i1)
              llg(i1+ntime) = nu(i1) - (y(i1,2)-y(i1,1)).*la1(se(i1)).*eta2(i1);
            else
              llg(i1+ntime) = nu(i1) - (y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1((sb(i1)+1):(se(i1)-1)).*eta2(i1)) - (lik.stime(sb(i1)+1)-y(i1,1)).*la1(sb(i1)).*eta2(i1);
            end
          end
        end
        
      end
      
    end
  end

  function [llg2,llg2mat] = lik_coxph_llg2(lik, y, f, param, z)
  %function [pi_vec, pi_mat] = lik_coxph_llg2(lik, y, ff, param, z)
  %LIK_COXPH_LLG2  Second gradients of the log likelihood
  %
  %  Description        
  %    LLG2 = LIK_COXPH_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z, and
  %    latent values F. Returns the Hessian of the log likelihood
  %    with respect to PARAM. At the moment PARAM can be only
  %    'latent'. LLG2 is a vector with diagonal elements of the
  %    Hessian matrix (off diagonals are zero).
  %
  %  See also
  %    LIK_COXPH_LL, LIK_COXPH_LLG, LIK_COXPH_LLG3, GPLA_E

    if isempty(z)
      error(['lik_coxph -> lik_coxph_llg2: missing z!   '... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input z. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end
    
    ntime=size(lik.xtime,1);
    [n,ny]=size(y);
    f1=f(1:ntime);
    f2=f((ntime+1):(ntime+n));
    
    la1=exp(f1);
    eta2=exp(f2);
    
    %nu=1-z;
    sd=lik.stime(2)-lik.stime(1);
    
    switch param
      case 'latent'
        
        if ny==1
          llg2=zeros(ntime+n,1);
          llg2mat=zeros(ntime,n);
          % 11
          for i1=1:ntime
            ind=y>=lik.stime(i1) & y<lik.stime(i1+1);
            llg2(i1)= sum(-(y(ind)-lik.stime(i1)).*la1(i1).*eta2(ind)) - sum(sum(sd.*la1(i1)).*eta2(~ind & y>=lik.stime(i1+1)));
            %llg2(i1,i1)= sum(-(y(ind)-lik.stime(i1)).*la1(i1).*eta2(ind)) - sum(sum(sd.*la1(i1)).*eta2(~ind & y>=lik.stime(i1+1)));
          end
          
          % 22
          for i1=1:n
            si=sum(y(i1)>lik.stime);
            llg2(i1+ntime)= -(y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
            %llg2(i1+ntime,i1+ntime)= -(y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
          end
          
          % derivative wrt f1 and f2:
          for i1=1:ntime
            ind=y>=lik.stime(i1) & y<lik.stime(i1+1);
            llg2mat(i1,find(ind))= -(y(ind)-lik.stime(i1)).*la1(i1).*eta2(ind);
            llg2mat(i1,find((~ind & y>=lik.stime(i1+1)))) = - sd.*la1(i1).*eta2((~ind & y>=lik.stime(i1+1)));
            %llg2(i1,ntime+find(ind))= -(y(ind)-lik.stime(i1)).*la1(i1).*eta2(ind);
            %llg2(i1,ntime+find((~ind & y>=lik.stime(i1+1)))) = - sd.*la1(i1).*eta2((~ind & y>=lik.stime(i1+1)));
            %llg2(ntime+find(ind),i1)=llg2(i1,ntime+find(ind));
            %llg2(ntime+find((~ind & y>=lik.stime(i1+1))),i1)=llg2(i1,ntime+find((~ind & y>=lik.stime(i1+1))));
          end
          
        else
          llg2=zeros(ntime+n,1);
          llg2mat=zeros(ntime,n);
          
          % 11
          for i1=1:ntime
            
            % left truncated + follow-up entry: (1)
            ind_vkst = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1+1);
            % follow-up entry + follow-up exit: (2)
            ind_stsp = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
            % follow-up: (3)
            ind_s = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1+1);
            % follow-up exit: (4)
            ind_sp = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
            
            
            % (1)
            s2b=sum(-(lik.stime(i1+1)-y(ind_vkst,1)).*la1(i1).*eta2(ind_vkst));
            % (2)
            s3b=-sum((y(ind_stsp,2)-y(ind_stsp,1)).*la1(i1).*eta2(ind_stsp));
            % (3)
            s4=-sum(sd.*la1(i1).*eta2(ind_s));
            % (4)
            s5=-sum((y(ind_sp,2)-lik.stime(i1)).*la1(i1).*eta2(ind_sp));
            llg2(i1) = s2b+s3b+s4+s5;
            %llg2(i1,i1) = s2b+s3b+s4+s5;
          end
          
          % 22
          sb=sum(bsxfun(@gt,y(:,1),lik.stime),2);
          se=sum(bsxfun(@gt,y(:,2),lik.stime),2);
          for i1=1:n
            %sb=sum(y(i1,1)>lik.stime); % begin
            %se=sum(y(i1,2)>lik.stime); % end
            
            if sb(i1)==0
              llg2(i1+ntime)= -(y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1(1:(se(i1)-1)).*eta2(i1));
              %llg2(i1+ntime,i1+ntime)= -(y(i1,2)-lik.stime(se)).*la1(se).*eta2(i1) - sum(sd.*la1(1:(se-1)).*eta2(i1));
            else
              if se(i1)==sb(i1)
                llg2(i1+ntime) = -(y(i1,2)-y(i1,1)).*la1(se(i1)).*eta2(i1);
                %llg2(i1+ntime,i1+ntime) = -(y(i1,2)-y(i1,1)).*la1(se).*eta2(i1);
              else
                llg2(i1+ntime) = -(y(i1,2)-lik.stime(se(i1))).*la1(se(i1)).*eta2(i1) - sum(sd.*la1((sb(i1)+1):(se(i1)-1)).*eta2(i1)) - (lik.stime(sb(i1)+1)-y(i1,1)).*la1(sb(i1)).*eta2(i1);
                %llg2(i1+ntime,i1+ntime) = -(y(i1,2)-lik.stime(se)).*la1(se).*eta2(i1) - sum(sd.*la1((sb+1):(se-1)).*eta2(i1)) - (lik.stime(sb+1)-y(i1,1)).*la1(sb).*eta2(i1);
              end
            end
          end
          
          % derivative wrt f1 and f2:
          for i1=1:ntime
            
            % left truncated + follow-up entry: (1)
            ind_vkst = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1+1);
            % follow-up entry + follow-up exit: (2)
            ind_stsp = y(:,1)>=lik.stime(i1) & y(:,1)<lik.stime(i1+1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
            % follow-up: (3)
            ind_s = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1+1);
            % follow-up exit: (4)
            ind_sp = y(:,1)<lik.stime(i1) & y(:,2)>=lik.stime(i1) & y(:,2)<lik.stime(i1+1);
            % (1)
            %llg2mat(i1,find(ind_vkst))=-(lik.stime(i1+1)-y(ind_vkst,1)).*la1(i1).*eta2(ind_vkst);
            llg2mat(i1,ind_vkst)=-(lik.stime(i1+1)-y(ind_vkst,1)).*la1(i1).*eta2(ind_vkst);
            %llg2mat(find(ind_vkst),i1)=llg2mat(i1,find(ind_vkst));
            
            % (2)
            %llg2mat(i1,find(ind_stsp))=-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(i1).*eta2(ind_stsp);
            llg2mat(i1,ind_stsp)=-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(i1).*eta2(ind_stsp);
            %llg2mat(find(ind_stsp),i1)=llg2mat(i1,find(ind_stsp));
            % (3)
            %llg2mat(i1,find(ind_s))= -sd.*la1(i1).*eta2(ind_s);
            llg2mat(i1,ind_s)= -sd.*la1(i1).*eta2(ind_s);
            %llg2mat(find(ind_s),i1)=llg2mat(i1,find(ind_s));
            % (4)
            %llg2mat(i1,find(ind_sp))=-(y(ind_sp,2)-lik.stime(i1)).*la1(i1).*eta2(ind_sp);
            llg2mat(i1,ind_sp)=-(y(ind_sp,2)-lik.stime(i1)).*la1(i1).*eta2(ind_sp);
            %llg2mat(find(ind_sp),i1)=llg2mat(i1,find(ind_sp));
          end
      end
    end
  end    
  
  function [llg3,llg3mat] = lik_coxph_llg3(lik, y, f, param, z, j1)
  %LIK_COXPH_LLG3  Third gradients of the log likelihood
  %
  %  Description
  %    LLG3 = LIK_COXPH_LLG3(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z and
  %    latent values F and returns the third gradients of the log
  %    likelihood with respect to PARAM. At the moment PARAM can be
  %    only 'latent'. LLG3 is a vector with third gradients.
  %
  %  See also
  %    LIK_COXPH_LL, LIK_COXPH_LLG, LIK_COXPH_LLG2, GPLA_E, GPLA_G

    if isempty(z)
      error(['lik_coxph -> lik_coxph_llg3: missing z!   '... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input z. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end

    ntime=size(lik.xtime,1);
    
    [n,ny]=size(y);
    f1=f(1:ntime);
    f2=f((ntime+1):(ntime+n));
    
    la1=exp(f1);
    eta2=exp(f2);
    
    %nu=1-z;
    sd=lik.stime(2)-lik.stime(1);
    
    switch param
      case 'latent'
        
        if ny==1
          %llg3=sparse(ntime+n,ntime+n);
          %llg3=zeros(ntime+n,ntime+n);
          llg3=zeros(ntime+n,1);
          llg3mat=zeros(ntime,n);
          
          if j1<=ntime
            
            % 11
            ind=y>=lik.stime(j1) & y<lik.stime(j1+1);
            fi=find(ind);
            fni=find(~ind & y>=lik.stime(j1+1));
            
            llg3(j1) = sum(-(y(ind)-lik.stime(j1)).*la1(j1).*eta2(ind)) - sum(sum(sd.*la1(j1)).*eta2(~ind & y>=lik.stime(j1+1)));
            %llg3(j1,j1) = sum(-(y(ind)-lik.stime(j1)).*la1(j1).*eta2(ind)) - sum(sum(sd.*la1(j1)).*eta2(~ind & y>=lik.stime(j1+1)));
            
            % 22
            %              llg3(ntime+fi,ntime+fi) = diag(-(y(ind)-lik.stime(j1)).*la1(j1).*eta2(ind));
            %              llg3(ntime+fni,ntime+fni) = diag(-sd.*la1(j1).*eta2((~ind & y>=lik.stime(j1+1))));
            
            if ~isempty(fi)
              valtmp=(-(y(ind)-lik.stime(j1)).*la1(j1).*eta2(ind));
              for m2i=1:length(valtmp)
                llg3( ntime+fi(m2i))  = valtmp(m2i);
                %llg3( ntime+fi(m2i), ntime+fi(m2i))  = valtmp(m2i);
              end
            end
            if ~isempty(fni)
              valtmp2=(-sd.*la1(j1).*eta2((~ind & y>=lik.stime(j1+1))));
              for m2i=1:length(valtmp2)
                llg3( ntime+fni(m2i))  = valtmp2(m2i);
                %llg3( ntime+fni(m2i), ntime+fni(m2i))  = valtmp2(m2i);
              end
            end
            
            % 12/21
            % derivative wrt f1 and f2:
            val1tmp=-(y(ind)-lik.stime(j1)).*la1(j1).*eta2(ind);
            llg3mat(j1,fi)= val1tmp;
            %llg3(j1,ntime+fi)= val1tmp;
            %llg3(ntime+fi,j1)=val1tmp;
            
            val2tmp = - sd.*la1(j1).*eta2((~ind & y>=lik.stime(j1+1)));
            llg3mat(j1,fni) = val2tmp;
            %llg3(j1,ntime+fni) = val2tmp;
            %llg3(ntime+fni,j1)=val2tmp;
            
          else
            
            % 11
            s1=sum(y(j1-ntime)>lik.stime);
            llg3(1:(s1-1)) = - sd.*la1(1:(s1-1)).*eta2(j1-ntime);
            llg3(s1) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime);
            %llg3(1:(s1-1),1:(s1-1)) = diag( - sd.*la1(1:(s1-1)).*eta2(j1-ntime));
            %llg3(s1,s1) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime);
            
            % 22
            llg3(j1) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime) - sum(sd.*la1(1:(s1-1)).*eta2(j1-ntime));
            %llg3(j1,j1) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime) - sum(sd.*la1(1:(s1-1)).*eta2(j1-ntime));
            
            % 12/21
            % derivative wrt f1 and f2:
            val3tmp = - sd.*la1(1:(s1-1)).*eta2(j1-ntime);
            llg3mat(1:(s1-1),j1-ntime)= val3tmp;
            %llg3(1:(s1-1),j1)= val3tmp;
            %llg3(j1,1:(s1-1))=val3tmp;
            
            llg3mat(s1,j1-ntime) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime);
            %llg3(s1,j1) = -(y(j1-ntime)-lik.stime(s1)).*la1(s1).*eta2(j1-ntime);
            %llg3(j1,s1)=llg3(s1,j1);
            
          end
          
        else
          llg3=zeros(ntime+n,1);
          llg3mat=zeros(ntime,n);
          
          if j1<=ntime
            
            % 11
            % left truncated + follow-up entry: (1)
            ind_vkst = y(:,1)>=lik.stime(j1) & y(:,1)<lik.stime(j1+1) & y(:,2)>=lik.stime(j1+1);
            % follow-up entry + follow-up exit: (2)
            ind_stsp = y(:,1)>=lik.stime(j1) & y(:,1)<lik.stime(j1+1) & y(:,2)>=lik.stime(j1) & y(:,2)<lik.stime(j1+1);
            % follow-up: (3)
            ind_s = y(:,1)<lik.stime(j1) & y(:,2)>=lik.stime(j1+1);
            % follow-up exit: (4)
            ind_sp = y(:,1)<lik.stime(j1) & y(:,2)>=lik.stime(j1) & y(:,2)<lik.stime(j1+1);
            
            % (1)
            s2b=sum(-(lik.stime(j1+1)-y(ind_vkst,1)).*la1(j1).*eta2(ind_vkst));
            % (2)
            s3b=-sum((y(ind_stsp,2)-y(ind_stsp,1)).*la1(j1).*eta2(ind_stsp));
            % (3)
            s4=-sum(sd.*la1(j1).*eta2(ind_s));
            % (4)
            s5=-sum((y(ind_sp,2)-lik.stime(j1)).*la1(j1).*eta2(ind_sp));
            
            llg3(j1) = s2b+s3b+s4+s5;
            %llg3(j1,j1) = s2b+s3b+s4+s5;
            
            % 22
            % (1)
            llg3(ntime+find(ind_vkst))=-(lik.stime(j1+1)-y(ind_vkst,1)).*la1(j1).*eta2(ind_vkst);
            %llg3(ntime+find(ind_vkst),ntime+find(ind_vkst))=diag(-(lik.stime(j1+1)-y(ind_vkst,1)).*la1(j1).*eta2(ind_vkst));
            % (2)
            llg3(ntime+find(ind_stsp))=-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(j1).*eta2(ind_stsp);
            %llg3(ntime+find(ind_stsp),ntime+find(ind_stsp))=diag(-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(j1).*eta2(ind_stsp));
            % (3)
            llg3(ntime+find(ind_s))= -sd.*la1(j1).*eta2(ind_s);
            %llg3(ntime+find(ind_s),ntime+find(ind_s))= diag(-sd.*la1(j1).*eta2(ind_s));
            % (4)
            llg3(ntime+find(ind_sp))=-(y(ind_sp,2)-lik.stime(j1)).*la1(j1).*eta2(ind_sp);
            %llg3(ntime+find(ind_sp),ntime+find(ind_sp))=diag(-(y(ind_sp,2)-lik.stime(j1)).*la1(j1).*eta2(ind_sp));
            
            % 12/21
            llg3mat(j1,find(ind_vkst))=-(lik.stime(j1+1)-y(ind_vkst,1)).*la1(j1).*eta2(ind_vkst);
            %llg3(j1,ntime+find(ind_vkst))=-(lik.stime(j1+1)-y(ind_vkst,1)).*la1(j1).*eta2(ind_vkst);
            %llg3(ntime+find(ind_vkst),j1)=llg3(j1,ntime+find(ind_vkst));
            % (2)
            llg3mat(j1,find(ind_stsp))=-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(j1).*eta2(ind_stsp);
            %llg3(j1,ntime+find(ind_stsp))=-(y(ind_stsp,2)-y(ind_stsp,1)).*la1(j1).*eta2(ind_stsp);
            %llg3(ntime+find(ind_stsp),j1)=llg3(j1,ntime+find(ind_stsp));
            % (3)
            llg3mat(j1,find(ind_s))= -sd.*la1(j1).*eta2(ind_s);
            %llg3(j1,ntime+find(ind_s))= -sd.*la1(j1).*eta2(ind_s);
            %llg3(ntime+find(ind_s),j1)=llg3(j1,ntime+find(ind_s));
            % (4)
            llg3mat(j1,find(ind_sp))=-(y(ind_sp,2)-lik.stime(j1)).*la1(j1).*eta2(ind_sp);
            %llg3(j1,ntime+find(ind_sp))=-(y(ind_sp,2)-lik.stime(j1)).*la1(j1).*eta2(ind_sp);
            %llg3(ntime+find(ind_sp),j1)=llg3(j1,ntime+find(ind_sp));
          else
            
            sb=sum(y(j1-ntime,1)>lik.stime); % begin
            se=sum(y(j1-ntime,2)>lik.stime); % end
            
            % 11
            if sb==0
              llg3(1:(se-1))= -sd.*la1(1:(se-1)).*eta2(j1-ntime);
              llg3(se)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
              %llg3(1:(se-1),1:(se-1))= diag( -sd.*la1(1:(se-1)).*eta2(j1-ntime));
              %llg3(se,se)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
            else
              if se==sb
                llg3(se) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
                %llg3(se,se) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
              else
                llg3(sb) = - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
                llg3((sb+1):(se-1)) = - sd.*la1((sb+1):(se-1)).*eta2(j1-ntime);
                llg3(se) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
                %llg3(sb,sb) = - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
                %llg3((sb+1):(se-1),(sb+1):(se-1)) = diag(- sd.*la1((sb+1):(se-1)).*eta2(j1-ntime));
                %llg3(se,se) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
              end
            end
            
            % 12/21
            if sb==0
              llg3mat(1:(se-1),j1-ntime) = -sd.*la1(1:(se-1)).*eta2(j1-ntime);
              %llg3(1:(se-1),j1) = -sd.*la1(1:(se-1)).*eta2(j1-ntime);
              %llg3(j1,1:(se-1)) = llg3(1:(se-1),j1);
              
              llg3mat(se,j1-ntime)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
              %llg3(se,j1)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
              %llg3(j1,se) = llg3(se,j1);
            else
              if se==sb
                llg3mat(se,j1-ntime) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
                %llg3(se,j1) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
                %llg3(j1,se) = llg3(se,j1);
              else
                llg3mat(sb,j1-ntime) = - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
                llg3mat((sb+1):(se-1),j1-ntime) = - sd.*la1((sb+1):(se-1)).*eta2(j1-ntime);
                llg3mat(se,j1-ntime) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
                %llg3(sb,j1) = - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
                %llg3(j1,sb) = llg3(sb,j1);
                %llg3((sb+1):(se-1),j1) = - sd.*la1((sb+1):(se-1)).*eta2(j1-ntime);
                %llg3(j1,(sb+1):(se-1)) = llg3((sb+1):(se-1),j1);
                %llg3(se,j1) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime);
                %llg3(j1,se) = llg3(se,j1);
              end
            end
            
            % 22
            if sb==0
              llg3(j1)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime) - sum(sd.*la1(1:(se-1)).*eta2(j1-ntime));
              %llg3(j1,j1)= -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime) - sum(sd.*la1(1:(se-1)).*eta2(j1-ntime));
            else
              if se==sb
                llg3(j1) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
                %llg3(j1,j1) = -(y(j1-ntime,2)-y(j1-ntime,1)).*la1(se).*eta2(j1-ntime);
              else
                llg3(j1) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime) - sum(sd.*la1((sb+1):(se-1)).*eta2(j1-ntime)) - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
                %llg3(j1,j1) = -(y(j1-ntime,2)-lik.stime(se)).*la1(se).*eta2(j1-ntime) - sum(sd.*la1((sb+1):(se-1)).*eta2(j1-ntime)) - (lik.stime(sb+1)-y(j1-ntime,1)).*la1(sb).*eta2(j1-ntime);
              end
            end
            
          end
        end
    end
  end

  
  function [m_0, m_1, sigm2hati1] = lik_coxph_tiltedMoments(lik, y, i1, S2_i, M_i, z)
      
      [n,ny]=size(y);
      
      % M_i(end);
      % S2_i(end,end);
      fgrid=M_i(end)+sqrt(S2_i(end,end))*[-6 6];
      fg=linspace(fgrid(1),fgrid(2),15);
      ng=length(fg);
      
      ntime=size(lik.xtime,1);
      
      %f11=f(1:ntime);
      %f2=f((ntime+1):(ntime+n));
      %la1=exp(f1);
      %eta2=exp(f2);
      
      nu=1-z;
      sd=lik.stime(2)-lik.stime(1);
      if ny==1
        sb=1;
        se=sum(bsxfun(@gt,y(i1,1),lik.stime),2);
      end
      indf=sb:se;
      sdvec=[ones(se-1,1)*sd; y(i1)-lik.stime(se)];
      nutmp=zeros(se,1);
      nutmp(se)=nu(i1);
      
      for j1=1:ng
        
        % conditional distribution
        myy=M_i(indf)+S2_i(indf,end)*(1./S2_i(end,end))*(fg(j1)-M_i(end));
        myy0=myy;
        Sigm=S2_i(indf,indf)-S2_i(indf,end)*(1./S2_i(end,end))*S2_i(end,indf);
        Sigm0=Sigm;
        
        nu_prior=Sigm\myy;
        
        nt=size(myy,1);
        c1=exp(fg(j1));
        
        % site parameters
        tautilde=zeros(nt,1);
        nutilde=zeros(nt,1);
        ztilde=zeros(nt,1);
        
        max_small_ep_iter=50;
        tol=1e-9;
        small_ep_iter=1;
        
        tautilde0=Inf; nutilde0=Inf; ztilde0=Inf;
        
        logZep_tmp=0; logZep=Inf;
        %while small_ep_iter <= max_small_ep_iter && (sum(abs(tautilde0-tautilde)>tol) || sum(abs(nutilde0-nutilde)>tol) || sum(abs(ztilde0-ztilde)>tol))
        while small_ep_iter<=max_small_ep_iter && abs(logZep_tmp-logZep)>tol
          logZep_tmp=logZep;
          
          
          %tautilde0=tautilde; nutilde0=nutilde; ztilde0=ztilde;
          
          for k1=1:nt
            
            tau_i=Sigm(k1,k1)^-1-tautilde(k1);
            nu_i = Sigm(k1,k1)^-1*myy(k1)-nutilde(k1);
            myy_i=nu_i/tau_i;
            sigm2_i=tau_i^-1;
            
            % marginal moments
            [M0(k1), muhati, sigm2hati] = coxph_tiltedMoments(sigm2_i, myy_i, nutmp(k1), sdvec(k1), c1);
            %[M0, muhati, sigm2hati] = coxph_tiltedMoments(lik, y(i1,:), k1, sigm2_i, myy_i, c1, sd_vec(i1), ztmp);
            
            deltatautilde=sigm2hati^-1-tau_i-tautilde(k1);
            tautilde(k1)=tautilde(k1)+deltatautilde;
            nutilde(k1)=sigm2hati^-1*muhati-nu_i;
            
            apu = deltatautilde/(1+deltatautilde*Sigm(k1,k1));
            Sigm = Sigm - apu*(Sigm(:,k1)*Sigm(:,k1)');
            
            % The below is how Rasmussen and Williams
            % (2006) do the update. The above version is
            % more robust.
            %apu = deltatautilde^-1+Sigm(k1,k1);
            %apu = (Sigm(:,k1)/apu)*Sigm(:,k1)';
            %Sigm = Sigm - apu;
            %Sigm=Sigm-(deltatautilde^-1+Sigm(k1,k1))^-1*(Sigm(:,k1)*Sigm(:,k1)');
            
            %myy=Sigm*nutilde;
            myy=Sigm*(nutilde+nu_prior);
            
            muvec_i(k1,1)=myy_i;
            sigm2vec_i(k1,1)=sigm2_i;
            
          end
          
          
          if tautilde > 0
            Stilde=tautilde;
            Stildesqroot=diag(sqrt(tautilde));
            B=eye(nt)+Stildesqroot*Sigm0*Stildesqroot;
            L=chol(B,'lower');
            
            V=(L\Stildesqroot)*Sigm0;
            Sigm=Sigm0-V'*V;
            %myy=Sigm*nutilde;
            myy=Sigm*(nutilde+nu_prior);
            
            %Ls = chol(Sigm);
            
            % Compute the marginal likelihood
            % Direct formula (3.65):
            % Sigmtilde=diag(1./tautilde);
            % mutilde=inv(Stilde)*nutilde;
            %
            % logZep=-0.5*log(det(Sigmtilde+K))-0.5*mutilde'*inv(K+Sigmtilde)*mutilde+
            %         sum(log(normcdf(y.*muvec_i./sqrt(1+sigm2vec_i))))+
            %         0.5*sum(log(sigm2vec_i+1./tautilde))+
            %         sum((muvec_i-mutilde).^2./(2*(sigm2vec_i+1./tautilde)))
            
            % 4. term & 1. term
            term41=0.5*sum(log(1+tautilde.*sigm2vec_i))-sum(log(diag(L)));
            
            % 5. term (1/2 element) & 2. term
            T=1./sigm2vec_i;
            Cnutilde = Sigm0*nutilde;
            L2 = V*nutilde;
            term52 = nutilde'*Cnutilde - L2'*L2 - (nutilde'./(T+Stilde)')*nutilde;
            term52 = term52.*0.5;
            
            % 5. term (2/2 element)
            term5=0.5*muvec_i'.*(T./(Stilde+T))'*(Stilde.*muvec_i-2*nutilde);
            
            % 3. term
            term3 = sum(log(M0));
            
            V_tmp=(L\Stildesqroot);
            Sigm_inv_tmp=V_tmp'*V_tmp;
            
            term_add1=-0.5*myy0'*Sigm_inv_tmp*myy0;
            term_add2=myy0'*(eye(nt)-Sigm_inv_tmp*Sigm0)*nutilde;
            logZep = -(term41+term52+term5+term3+term_add1+term_add2);
            
            %logZep = -(term41+term52+term5+term3);
            
            small_ep_iter=small_ep_iter+1;
            %iter=iter+1;
          else
            error('tautilde <= 0')
          end
        end
        
        ZZ(j1,1)=exp(-logZep);
        MM(:,j1)=myy;
        SS2(:,:,j1)=Sigm;
        
      end
      
      %m_0=zeros(1,1);
      %m_1=zeros(nt+1,1);
      %sigm2hati1=zeros(nt+1,nt+1);
      % indf
      
      W=normpdf(fg,M_i(end),sqrt(S2_i(end,end)))*(fg(2)-fg(1));
      
      qw=W.*ZZ';
      m_0=sum(qw);
      m_1=[sum(bsxfun(@times,qw,MM),2); sum(qw.*fg)]./m_0;
      
      m_211=zeros(nt,nt);
      for k1=1:ng
        m_211=m_211+qw(k1)*(SS2(:,:,k1)+MM(:,k1)*MM(:,k1)');
      end
      m_212=(qw.*fg)*MM';
      m_222=(qw.*fg)*fg';
      
      m_2=[m_211 m_212'; m_212 m_222]./m_0;
      
      sigm2hati1=m_2 - m_1*m_1';
      
      %figure(1),hold on, plot(fg(j1),logZep,'.')
      %figure(2),hold on, plot(fg(j1),exp(-logZep),'.')
      
  end


%   function [m_0, m_1, sigm2hati1] = lik_coxph_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
%   %LIK_COXPH_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%   %
%   %  Description
%   %    [M_0, M_1, M2] = LIK_COXPH_TILTEDMOMENTS(LIK, Y, I, S2,
%   %    MYY, Z) takes a likelihood structure LIK, incedence counts
%   %    Y, expected counts Z, index I and cavity variance S2 and
%   %    mean MYY. Returns the zeroth moment M_0, mean M_1 and
%   %    variance M_2 of the posterior marginal (see Rasmussen and
%   %    Williams (2006): Gaussian processes for Machine Learning,
%   %    page 55).
%   %
%   %  See also
%   %    GPEP_E
%     
%     if isempty(z)
%       error(['lik_coxph -> lik_coxph_tiltedMoments: missing z!'... 
%              'Coxph likelihood needs the expected number of            '...
%              'occurrences as an extra input z. See, for                 '...
%              'example, lik_coxph and gpep_e.                       ']);
%     end
%     
%     yy = y(i1);
%     avgE = z(i1);
%     
%     % get a function handle of an unnormalized tilted distribution 
%     % (likelihood * cavity = Negative-binomial * Gaussian)
%     % and useful integration limits
%     [tf,minf,maxf]=init_coxph_norm(yy,myy_i,sigm2_i,avgE,r);
%     
%     % Integrate with quadrature
%     RTOL = 1.e-6;
%     ATOL = 1.e-10;
%     [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
%     sigm2hati1 = m_2 - m_1.^2;
%     
%     % If the second central moment is less than cavity variance
%     % integrate more precisely. Theoretically for log-concave
%     % likelihood should be sigm2hati1 < sigm2_i.
%     if sigm2hati1 >= sigm2_i
%       ATOL = ATOL.^2;
%       RTOL = RTOL.^2;
%       [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
%       sigm2hati1 = m_2 - m_1.^2;
%       if sigm2hati1 >= sigm2_i
%         error('lik_coxph_tilted_moments: sigm2hati1 >= sigm2_i');
%       end
%     end
%     
%   end
  
  function [lpyt,Ey, Vary] = lik_coxph_predy(lik, Ef, Covf, yt, zt)
  %LIK_COXPH_PREDY  Returns the predictive mean, variance and density of y
  %
  %  Description         
  %    [EY, VARY] = LIK_COXPH_PREDY(LIK, EF, VARF) takes a
  %    likelihood structure LIK, posterior mean EF and posterior
  %    Variance VARF of the latent variable and returns the
  %    posterior predictive mean EY and variance VARY of the
  %    observations related to the latent variables
  %        
  %    [Ey, Vary, PY] = LIK_COXPH_PREDY(LIK, EF, VARF YT, ZT)
  %    Returns also the predictive density of YT, that is 
  %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
  %    This requires also the incedence counts YT, expected counts ZT.
  %
  %  See also
  %    GPLA_PRED, GPEP_PRED, GPMC_PRED

    if isempty(zt)
      error(['lik_coxph -> lik_coxph_predy: missing zt!'... 
             'Coxph likelihood needs the expected number of    '...
             'occurrences as an extra input zt. See, for         '...
             'example, lik_coxph and gpla_e.               ']);
    end
    
% %
%     n=size(y,1);
%     f1=f(1:ntime);
%     f2=f((ntime+1):(ntime+n));
%     
%     la1=exp(f1);
%     eta2=exp(f2);
%     
%     nu=1-z;
%     sd=lik.stime(2)-lik.stime(1);
% 
%     ll=0;
%     for i1=1:n
%         si=sum(y(i1)>lik.stime);
%         ll=ll + nu(i1).*(f1(si)+f2(i1)) - (y(i1)-lik.stime(si)).*la1(si).*eta2(i1) - sum(sd.*la1(1:(si-1)).*eta2(i1));
%     end
% &
    ntime=size(lik.xtime,1);
    ntest=size(zt,1);
        
    Py = zeros(size(zt));
    %Ey = zeros(size(zt));
    %EVary = zeros(size(zt));
    %VarEy = zeros(size(zt));
    
    S=10000;
    for i1=1:ntest
      Sigm_tmp=Covf([1:ntime i1+ntime],[1:ntime i1+ntime]);
      Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
      f_star=mvnrnd(Ef([1:ntime i1+ntime]), Sigm_tmp, S);
      
      
      f1=f_star(:,1:ntime);
      f2=f_star(:,(ntime+1):end);
      
      la1=exp(f1);
      eta2=exp(f2);
      nu=1-zt;
      sd=lik.stime(2)-lik.stime(1);
      
      si=sum(yt(i1)>lik.stime);
      Py(i1)=mean(exp(nu(i1).*(f1(:,si)+f2) - (yt(i1)-lik.stime(si)).*la1(:,si).*eta2 - sum(sd.*la1(:,1:(si-1)),2).*eta2));
      
    end
    Ey = [];
    Vary = [];
    lpyt=log(Py);
    
    %     % Evaluate Ey and Vary
%     for i1=1:length(Ef)
%       %%% With quadrature
%       myy_i = Ef(i1);
%       sigm_i = sqrt(Varf(i1));
%       minf=myy_i-6*sigm_i;
%       maxf=myy_i+6*sigm_i;
% 
%       F = @(f) exp(log(avgE(i1))+f+norm_lpdf(f,myy_i,sigm_i));
%       Ey(i1) = quadgk(F,minf,maxf);
%       
%       F2 = @(f) exp(log(avgE(i1).*exp(f)+((avgE(i1).*exp(f)).^2/r))+norm_lpdf(f,myy_i,sigm_i));
%       EVary(i1) = quadgk(F2,minf,maxf);
%       
%       F3 = @(f) exp(2*log(avgE(i1))+2*f+norm_lpdf(f,myy_i,sigm_i));
%       VarEy(i1) = quadgk(F3,minf,maxf) - Ey(i1).^2;
%     end
%     Vary = EVary + VarEy;
% 
%     % Evaluate the posterior predictive densities of the given observations
%     if nargout > 2
%       for i1=1:length(Ef)
%         % get a function handle of the likelihood times posterior
%         % (likelihood * posterior = Negative-binomial * Gaussian)
%         % and useful integration limits
%         [pdf,minf,maxf]=init_coxph_norm(...
%           yt(i1),Ef(i1),Varf(i1),avgE(i1),r);
%         % integrate over the f to get posterior predictive distribution
%         Py(i1) = quadgk(pdf, minf, maxf);
%       end
%     end
  end

  function [m_0, m_1, sigm2hati1] = coxph_tiltedMoments(sigm2_i, myy_i, nutmp, sd, c1)
  
  integrand = @(f) exp(-c1.*exp(f).*sd + nutmp*(f+log(c1)) - log(sigm2_i)/2 - log(2*pi)/2 - 0.5*(f-myy_i).^2./sigm2_i);
  RTOL = 1.e-6;
  ATOL = 1.e-10;
  minf=myy_i+sqrt(sigm2_i)*(-6);
  maxf=myy_i+sqrt(sigm2_i)*(6);
  
  [m_0, m_1, m_2] = quad_moments(integrand, minf, maxf, RTOL, ATOL);
  sigm2hati1 = m_2 - m_1.^2;
  
  % If the second central moment is less than cavity variance
  % integrate more precisely. Theoretically for log-concave
  % likelihood should be sigm2hati1 < sigm2_i.
  
  if sigm2hati1 >= sigm2_i
    ATOL = ATOL.^2;
    RTOL = RTOL.^2;
    [m_0, m_1, m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    sigm2hati1 = m_2 - m_1.^2;
    if sigm2hati1 >= sigm2_i
      error('lik_poisson_tilted_moments: sigm2hati1 >= sigm2_i');
    end
  end
  
  end


  function [df,minf,maxf] = init_coxph_norm(yy,myy_i,sigm2_i,avgE,r)
  %INIT_COXPH_NORM
  %
  %  Description
  %    Return function handle to a function evaluating
  %    Negative-Binomial * Gaussian which is used for evaluating
  %    (likelihood * cavity) or (likelihood * posterior) Return
  %    also useful limits for integration. This is private function
  %    for lik_coxph.
  %  
  %  See also
  %    LIK_COXPH_TILTEDMOMENTS, LIK_COXPH_SITEDERIV,
  %    LIK_COXPH_PREDY
    
  % avoid repetitive evaluation of constant part
    ldconst = -gammaln(r)-gammaln(yy+1)+gammaln(r+yy)...
              - log(sigm2_i)/2 - log(2*pi)/2;
    % Create function handle for the function to be integrated
    df = @coxph_norm;
    % use log to avoid underflow, and derivates for faster search
    ld = @log_coxph_norm;
    ldg = @log_coxph_norm_g;
    ldg2 = @log_coxph_norm_g2;

    % Set the limits for integration
    % Negative-binomial likelihood is log-concave so the coxph_norm
    % function is unimodal, which makes things easier
    if yy==0
      % with yy==0, the mode of the likelihood is not defined
      % use the mode of the Gaussian (cavity or posterior) as a first guess
      modef = myy_i;
    else
      % use precision weighted mean of the Gaussian approximation
      % of the Negative-Binomial likelihood and Gaussian
      mu=log(yy/avgE);
      s2=(yy+r)./(yy.*r);
      modef = (myy_i/sigm2_i + mu/s2)/(1/sigm2_i + 1/s2);
    end
    % find the mode of the integrand using Newton iterations
    % few iterations is enough, since the first guess in the right direction
    niter=4;       % number of Newton iterations
    mindelta=1e-6; % tolerance in stopping Newton iterations
    for ni=1:niter
      g=ldg(modef);
      h=ldg2(modef);
      delta=-g/h;
      modef=modef+delta;
      if abs(delta)<mindelta
        break
      end
    end
    % integrand limits based on Gaussian approximation at mode
    modes=sqrt(-1/h);
    minf=modef-8*modes;
    maxf=modef+8*modes;
    modeld=ld(modef);
    iter=0;
    % check that density at end points is low enough
    lddiff=20; % min difference in log-density between mode and end-points
    minld=ld(minf);
    step=1;
    while minld>(modeld-lddiff)
      minf=minf-step*modes;
      minld=ld(minf);
      iter=iter+1;
      step=step*2;
      if iter>100
        error(['lik_coxph -> init_coxph_norm: ' ...
               'integration interval minimun not found ' ...
               'even after looking hard!'])
      end
    end
    maxld=ld(maxf);
    step=1;
    while maxld>(modeld-lddiff)
      maxf=maxf+step*modes;
      maxld=ld(maxf);
      iter=iter+1;
      step=step*2;
      if iter>100
        error(['lik_coxph -> init_coxph_norm: ' ...
               'integration interval maximun not found ' ...
               'even after looking hard!'])
      end
    end
    
%     while minld>(modeld-lddiff)
%       minf=minf-modes;
%       minld=ld(minf);
%       iter=iter+1;
%       if iter>100
%         error(['lik_coxph -> init_coxph_norm: ' ...
%                'integration interval minimun not found ' ...
%                'even after looking hard!'])
%       end
%     end
%     maxld=ld(maxf);
%     while maxld>(modeld-lddiff)
%       maxf=maxf+modes;
%       maxld=ld(maxf);
%       iter=iter+1;
%       if iter>100
%         error(['lik_coxph -> init_coxph_norm: ' ...
%                'integration interval maximum not found ' ...
%                'even after looking hard!'])
%       end
%       
%     end
    
    function integrand = coxph_norm(f)
    % Negative-binomial * Gaussian
      mu = avgE.*exp(f);
      integrand = exp(ldconst ...
                      +yy.*(log(mu)-log(r+mu))+r.*(log(r)-log(r+mu)) ...
                      -0.5*(f-myy_i).^2./sigm2_i);
    end
    
    function log_int = log_coxph_norm(f)
    % log(Negative-binomial * Gaussian)
    % log_coxph_norm is used to avoid underflow when searching
    % integration interval
      mu = avgE.*exp(f);
      log_int = ldconst...
                +yy.*(log(mu)-log(r+mu))+r.*(log(r)-log(r+mu))...
                -0.5*(f-myy_i).^2./sigm2_i;
    end
    
    function g = log_coxph_norm_g(f)
    % d/df log(Negative-binomial * Gaussian)
    % derivative of log_coxph_norm
      mu = avgE.*exp(f);
      g = -(r.*(mu - yy))./(mu.*(mu + r)).*mu ...
          + (myy_i - f)./sigm2_i;
    end
    
    function g2 = log_coxph_norm_g2(f)
    % d^2/df^2 log(Negative-binomial * Gaussian)
    % second derivate of log_coxph_norm
      mu = avgE.*exp(f);
      g2 = -(r*(r + yy))/(mu + r)^2.*mu ...
           -1/sigm2_i;
    end
    
  end

  function p = lik_coxph_invlink(lik, f, z)
  %LIK_COXPH_INVLINK  Returns values of inverse link function
  %             
  %  Description 
  %    P = LIK_COXPH_INVLINK(LIK, F) takes a likelihood structure LIK and
  %    latent values F and returns the values of inverse link function P.
  %
  %     See also
  %     LIK_COXPH_LL, LIK_COXPH_PREDY
  
    p = exp(f);
  end
  
  function reclik = lik_coxph_recappend(reclik, ri, lik)
  %RECAPPEND  Append the parameters to the record
  %
  %  Description 
  %    RECLIK = GPCF_COXPH_RECAPPEND(RECLIK, RI, LIK) takes a
  %    likelihood record structure RECLIK, record index RI and
  %    likelihood structure LIK with the current MCMC samples of
  %    the parameters. Returns RECLIK which contains all the old
  %    samples and the current samples from LIK.
  % 
  %  See also
  %    GP_MC

  % Initialize record
    if nargin == 2
      reclik.type = 'Coxph';

      % Set the function handles
      reclik.fh.pak = @lik_coxph_pak;
      reclik.fh.unpak = @lik_coxph_unpak;
      reclik.fh.lp = @lik_coxph_lp;
      reclik.fh.lpg = @lik_coxph_lpg;
      reclik.fh.ll = @lik_coxph_ll;
      reclik.fh.llg = @lik_coxph_llg;    
      reclik.fh.llg2 = @lik_coxph_llg2;
      reclik.fh.llg3 = @lik_coxph_llg3;
      reclik.fh.tiltedMoments = @lik_coxph_tiltedMoments;
      reclik.fh.predy = @lik_coxph_predy;
      reclik.fh.invlink = @lik_coxph_invlink;
      reclik.fh.recappend = @lik_coxph_recappend;
      return
    end

  end
end

