function [apc_prcs, apc_mean, apcf_prcs, apcf_mean]=gp_avpredcomp(gp, x, y, varargin)
%GP_AVPREDCOMP  Average predictive comparison for Gaussian process model
%
%  Description
%    [APC_PRCS, APC_MEAN, APCF_PRCS, APCF_MEAN]=GP_AVPREDCOMP(GP, X, Y, OPTIONS)
%    Takes a Gaussian process structure GP together with a matrix X of
%    training inputs and vector Y of training targets, and returns average
%    predictive comparison estimates for each input. APC_PRCS and APC_MEAN
%    are estimated predictive relevances for percentiles and means for each
%    input variable when latent outcome variable is squashed through the
%    inverse link function. APCF_PRCS and APCF_MEAN are estimated
%    predictive relevances for percentiles and mean values for each input
%    variable when outcome variable is the latent variable. 
%
%    OPTIONS is optional parameter-value pair
%      z         - optional observed quantity in triplet (x_i,y_i,z_i)
%                  Some likelihoods may use this. For example, in case of 
%                  Poisson likelihood we have z_i=E_i, that is, expected value 
%                  for ith case. 
%      nsamp     - determines the number of samples used (default=500).
%      prctiles  - determines percentiles that are computed from 0 to 100
%                  (default=[2.5 97.5]).
%      absvalues - makes average predictive comparison using absolute
%                  values ('on') of differences in outcomes (default='off')
%
%  See also
%    GP_PRED

% Copyright (c) 2011      Jaakko Riihimäki
% Copyright (c) 2011      Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GP_AVPREDCOMP';
ip.addRequired('gp',@isstruct);
ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
ip.addParamValue('nsamp', 500, @(x) isreal(x) && isscalar(x))
ip.addParamValue('prctiles', [2.5 97.5], @(x) isreal(x) && isvector(x) && all(x>=0) && all(x<=100))
ip.addParamValue('absvalues', 'off', @(x) isempty(x) || (ischar(x) && strcmp(x, 'off')) || (ischar(x) && strcmp(x, 'on')))

ip.parse(gp, x, y, varargin{:});
z=ip.Results.z;
nsamp=ip.Results.nsamp;
prctiles=ip.Results.prctiles;
absvalues=ip.Results.absvalues;

[n, nin]=size(x);

apc_prcs=zeros(nin,length(prctiles));
apc_mean=zeros(nin,1);
apcf_prcs=zeros(nin,length(prctiles));
apcf_mean=zeros(nin,1);

covx=cov(x);

stream = RandStream('mrg32k3a');
prevstream=RandStream.setDefaultStream(stream);

% loop through the input variables
for k1=1:nin
    
    %- Compute the weight matrix based on Mahalanobis distances:
    x_=x; x_(:,k1)=[];
    covx_=covx; covx_(:,k1)=[]; covx_(k1,:)=[];
    % weight matrix:
    W=zeros(n);
    for i1=1:n
        x_diff=bsxfun(@minus,x_(i1,:),x_((i1+1):n,:))';
        W(i1,(i1+1):n)=1./(1+sum(x_diff.*(covx_\x_diff)));
    end
    W=W+W'+eye(n);
    %-
    
    rsubstream=round(rand*10e9);
    
    
    num=zeros(1,nsamp); den=0;
    numf=zeros(1,nsamp);
    for i1=1:n
        % inputs of interest
        ui=x(i1, k1);
        ujs=x(:, k1);
        
        % replicate same values for other inputs
        x_irepj=repmat(x(i1,:),n,1); x_irepj(:,k1)=ujs;
        
        % 
        Udiff=[ujs-ui];
        Usign=sign(Udiff);
        stream.Substream = rsubstream;
        fs = gp_rnd(gp, x, y, x_irepj,'nsamp',nsamp);
        
        % squashe latent values through the inverse link function
        ilfs = feval(gp.lik.fh.invlink, gp.lik, fs, z);
        
        Wrep=repmat(W(:,i1),1,nsamp);
        Usignrep=repmat(Usign,1,nsamp);
        
        if strcmp(absvalues, 'on')
            % average absolute change in outcome
            num=num+sum(Wrep.*abs((ilfs-repmat(ilfs(i1,:),n,1)).*Usignrep));
            % average absolute change in latent outcome
            numf=numf+sum(Wrep.*abs((fs-repmat(fs(i1,:),n,1)).*Usignrep));            
        else
            % average change in outcome
            num=num+sum(Wrep.*(ilfs-repmat(ilfs(i1,:),n,1)).*Usignrep);
            % average change in latent outcome
            numf=numf+sum(Wrep.*(fs-repmat(fs(i1,:),n,1)).*Usignrep);
        end
        % average change in input
        den=den+sum(W(:,i1).*Udiff.*Usign);
    end
    
    % percentiles and mean values when outcome is computed through the inverse
    % link function
    apc_prcs(k1,:)=prctile(num./den,prctiles);
    apc_mean(k1,:)=mean(num./den);    

    % percentiles and mean values when outcome is the latent function
    apcf_prcs(k1,:)=prctile(numf./den,prctiles);
    apcf_mean(k1,:)=mean(numf./den);

end

RandStream.setDefaultStream(prevstream);

