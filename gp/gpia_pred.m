function [Eft, Varft, Eyt, Varyt, pyt, ft, pft] = ia_pred(gp_array, x, y, xt, varargin) 
%IA_PRED  Prediction with Gaussian Process GP_IA solution.
%
%  Description
%    [EFT, VARFT, EYT, VARYT] = IA_PRED(GP_ARRAY, X, Y, XT,
%    OPTIONS) takes a cell array of GP structures together
%    with matrix X of training inputs and vector Y of training
%    targets, and evaluates the predictive distribution at test
%    inputs XT with parameters marginalized out with IA. 
%    Returns a posterior mean EFT and variance VARFT of latent
%    variables and the posterior predictive mean EYT and variance
%    VARYT.
%
%    [EFT, VARFT, EYT, VARYT, PYT] = IA_PRED(GP, X, Y, XT, 'yt', YT, ...)
%    returns also the predictive density PYT of the observations YT
%    at test input locations XT with parameters marginalized
%    out with IA. This can be used for example in the
%    cross-validation. Here Y has to be vector.
%
%    [EFT, VARFT, EYT, VARYT, PYT, FT, PFT] = ...
%      IA_PRED(GP_ARRAY, X, Y, XT, OPTIONS) 
%    returns also the numerical representation of the marginal
%    posterior of latent variables at each XT. FT is a vector of
%    latent values and PFT_i = p(FT_i) is the posterior density for
%    FT_i.
%
%    OPTIONS is optional parameter-value pair
%      predcf - index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). See
%               additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Deafult is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a vector
%               of length n that points out the test inputs that
%               are also in the training set (if none, set TSTIND=[])
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case
%               of Poisson likelihood we have z_i=E_i, that is,
%               expected value for ith case.
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case
%               of Poisson likelihood we have z_i=E_i, that is, the
%               expected value for the ith case.
%       
%    NOTE! In case of FIC and PIC sparse approximation the
%    prediction for only some PREDCF covariance functions is just
%    an approximation since the covariance functions are coupled in
%    the approximation and are not strictly speaking additive
%    anymore.
%
%    For example, if you use covariance such as K = K1 + K2 your
%    predictions Eft1 = ia_pred(gp_array, X, Y, X, 'predcf', 1) and
%    Eft2 = ia_pred(gp_array, x, y, x, 'predcf', 2) should sum up to
%    Eft = ia_pred(gp_array, x, y, x). That is Eft = Eft1 + Eft2. With
%    FULL model this is true but with FIC and PIC this is true only
%    approximately. That is Eft \approx Eft1 + Eft2.
%
%    With CS+FIC the predictions are exact if the PREDCF covariance
%    functions are all in the FIC part or if they are CS
%    covariances.
%
%    NOTE! When making predictions with a subset of covariance
%    functions with FIC approximation the predictive variance can
%    in some cases be ill-behaved i.e. negative or unrealistically
%    small. This may happen because of the approximative nature of
%    the prediction.
%
%  See also
%    GP_PRED, GP_SET, GP_IA
%
        
% Copyright (c) 2009 Ville Pietiläinen
% Copyright (c) 2009-2010 Jarno Vanhatalo    

% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.    

    
    ip=inputParser;
    ip.FunctionName = 'IA_PRED';
    ip.addRequired('gp_array', @iscell);
    ip.addRequired('x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addRequired('xt', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('yt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('z', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('zt', [], @(x) isreal(x) && all(isfinite(x(:))))
    ip.addParamValue('predcf', [], @(x) isempty(x) || ...
                     isvector(x) && isreal(x) && all(isfinite(x)&x>0))
    ip.addParamValue('tstind', [], @(x) isempty(x) || iscell(x) ||...
                     (isvector(x) && isreal(x) && all(isfinite(x)&x>0)))
    ip.parse(gp_array, x, y, xt, varargin{:});
    yt=ip.Results.yt;
    z=ip.Results.z;
    zt=ip.Results.zt;
    predcf=ip.Results.predcf;
    tstind=ip.Results.tstind;
    
    % pass these forward
    options=struct();
    if ~isempty(ip.Results.yt);options.yt=ip.Results.yt;end
    if ~isempty(ip.Results.z);options.z=ip.Results.z;end
    if ~isempty(ip.Results.zt);options.zt=ip.Results.zt;end
    if ~isempty(ip.Results.predcf);options.predcf=ip.Results.predcf;end
    if ~isempty(ip.Results.tstind);options.tstind=ip.Results.tstind;end
    
    if nargout > 4 && isempty(yt)
      pyt = NaN;
    end
        
    nGP = numel(gp_array);
    
    for i=1:nGP
        P_TH(i,:) = gp_array{i}.ia_weight;
    end

    % =======================================================
    % Select the functions corresponding to the latent_method
    % =======================================================

    if isfield(gp_array{1}, 'latent_method')
        switch gp_array{1}.latent_method
          case 'EP'
            fh_e = @gpep_e;
            fh_g = @gpep_g;
            fh_p = @ep_pred;
          case 'Laplace'
            fh_e = @gpla_e;
            fh_g = @gpla_g;
            fh_p = @la_pred;
        end
    else 
        fh_e = @gp_e;
        fh_g = @gp_g;
        fh_p = @gp_pred;
    end
    
    % ==================================================
    % Make predictions with different models in gp_array
    % ==================================================

    for j = 1:nGP
        if isempty(yt)
            [Eft_grid(j,:), Varft_grid(j,:), Eyt_grid(j,:), Varyt_grid(j,:)]=feval(fh_p,gp_array{j},x,y,xt,options);
        else
            [Eft_grid(j,:), Varft_grid(j,:), Eyt_grid(j,:), Varyt_grid(j,:), pyt_grid(j,:)]=feval(fh_p,gp_array{j},x,y,xt, options);
        end
    end
    
    % ====================================================================
    % Grid of 501 points around 10 stds to both directions around the mode
    % ====================================================================

    % ==============================
    % Latent variables f
    % ==============================
    
    ft = zeros(size(Eft_grid,2),501);
    for j = 1 : size(Eft_grid,2);
        ft(j,:) = Eft_grid(1,j)-10*sqrt(Varft_grid(1,j)) : 20*sqrt(Varft_grid(1,j))/500 : Eft_grid(1,j)+10*sqrt(Varft_grid(1,j));  
    end
    
    % Calculate the density in each grid point by integrating over
    % different models
    pft = zeros(size(Eft_grid,2),501);
    for j = 1 : size(Eft_grid,2)
        pft(j,:) = sum(normpdf(repmat(ft(j,:),size(Eft_grid,1),1), repmat(Eft_grid(:,j),1,size(ft,2)), repmat(sqrt(Varft_grid(:,j)),1,size(ft,2))).*repmat(P_TH,1,size(ft,2)),1); 
    end

    % Normalize distributions
    pft = bsxfun(@rdivide,pft,sum(pft,2));

    % Widths of each grid point
    dft = diff(ft,1,2);
    dft(:,end+1)=dft(:,end);

    % Calculate mean and variance of the distributions
    Eft = sum(ft.*pft,2)./sum(pft,2);
    Varft = sum(pft.*(repmat(Eft,1,size(ft,2))-ft).^2,2)./sum(pft,2);
    
    Eyt = sum(Eyt_grid.*repmat(P_TH,1,size(Eyt_grid,2)),1);
    Varyt = sum(Varyt_grid.*repmat(P_TH,1,size(Eyt_grid,2)),1) + sum((Eyt_grid - repmat(Eyt,nGP,1)).^2, 1);
    Eyt=Eyt';
    Varyt=Varyt';
    
    if ~isempty(yt)
      pyt = sum(bsxfun(@times,pyt_grid,P_TH),1)';
    end
    
