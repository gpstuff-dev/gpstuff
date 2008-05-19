function likelih = likelih_probit(do, varargin)
%likelih_probit	   Create a probit likelihood structure for Gaussian Process
%
%	Description
%
%	likelih = LIKELIH_PROBIT('INIT', NIN) Create and initialize squared exponential
%       covariance function fo Gaussian process
%
%	The fields and (default values) in LIKELIH_PROBIT are:
%	  type           = 'likelih_probit'
%
%	likelih = LIKELIH_PROBIT('SET', LIKELH, 'FIELD1', VALUE1, 'FIELD2', VALUE2, ...)
%       Set the values of fields FIELD1... to the values VALUE1... in LIKELIH.
%
%	See also
%
%
%

% Copyright (c) 2006-2007 Jarno Vanhatalo

% This software is distributed under the GNU General Public
% License (version 2 or later); please refer to the file
% License.txt, included with the software, for details.

    if nargin < 2
        error('Not enough arguments')
    end

    % Initialize the covariance function
    if strcmp(do, 'init')
        likelih.type = 'probit';
        
        % Set the function handles to the nested functions
        likelih.fh_pak = @likelih_probit_pak;
        likelih.fh_unpak = @likelih_probit_unpak;
        likelih.fh_permute = @likelih_probit_permute;
        likelih.fh_e = @likelih_probit_e;
        likelih.fh_g = @likelih_probit_g;    
        likelih.fh_hessian = @likelih_probit_hessian;
        likelih.fh_g3 = @likelih_probit_g3;
        likelih.fh_tiltedMoments = @likelih_probit_tiltedMoments;
        likelih.fh_mcmc = @likelih_probit_mcmc;
        likelih.fh_recappend = @likelih_probit_recappend;
        
        if length(varargin) > 1
            if mod(nargin,2) ~=0
                error('Wrong number of arguments')
            end
            % Loop through all the parameter values that are changed
            for i=2:2:length(varargin)-1
                switch varargin{i}
                  otherwise
                    error('Wrong parameter name!')
                end
            end
        end
    end

    % Set the parameter values of covariance function
    if strcmp(do, 'set')
        if mod(nargin,2) ~=0
            error('Wrong number of arguments')
        end
        gpcf = varargin{1};
        % Loop through all the parameter values that are changed
        for i=2:2:length(varargin)-1
            switch varargin{i}
              otherwise
                error('Wrong parameter name!')
            end
        end
    end



    function w = likelih_probit_pak(likelih, w)
    %
    %
    %   
    end


    function w = likelih_probit_unpak(likelih, w)
    %
    %
    %    
    end



    function likelih = likelih_probit_permute(likelih, p)
    %
    %
    %
    end


    function logLikelih = likelih_probit_e(likelih, y, f)
    %
    %
    %
        logLikelih = sum(log(normcdf(y.*f)));
    end


    function deriv = likelih_probit_g(likelih, y, f, param)
        switch param
          case 'latent'
            deriv = y.*normpdf(f)./normcdf(y.*f);
        end
    end


    function hessian = likelih_probit_hessian(likelih, y, f, param)
    %
    %
    %
        switch param
          case 'latent'
            z = y.*f;
            hessian = -(normpdf(f)./normcdf(z)).^2 - z.*normpdf(f)./normcdf(z);
        end
    end
    
    function thir_grad = likelih_probit_g3(likelih, y, f, param)
    %
    %
    %
        switch param
          case 'latent'
            z2 = normpdf(f)./normcdf(y.*f);
            thir_grad = 2.*y.*z2.^3 + 3.*f.*z2.^2 - z2.*(y-y.*f.^2);
        end
    end
    

    function [m_0, m_1, m_2] = likelih_probit_tiltedMoments(likelih, y, i1, sigm2_i, myy_i)
    %
    %
    %
        m_0 = normcdf(y(i1).*myy_i./sqrt(1+sigm2_i));
        zi=y(i1)*myy_i/sqrt(1+sigm2_i);
        normp_zi = normpdf(zi);
        normc_zi = normcdf(zi);
        muhati1=myy_i+(y(i1)*sigm2_i*normp_zi)/(normc_zi*sqrt(1+sigm2_i));
        sigm2hati1=sigm2_i-(sigm2_i^2*normp_zi)/((1+sigm2_i)*normc_zi)*(zi+normp_zi/normc_zi);
        m_1 = muhati1;
        m_2 = sigm2hati1;
    end


    function [z, energ, diagn] = likelih_probit_mcmc(z, opt, varargin)
    %
    %
    %
        
    end


    function reclikelih = likelih_probit_recappend(reclikelih, ri, likelih)
    % RECAPPEND - Record append
    %          Description
    %          RECCF = GPCF_SEXP_RECAPPEND(RECCF, RI, GPCF) takes old covariance
    %          function record RECCF, record index RI, RECAPPEND returns a
    %          structure RECCF containing following record fields:
    %          lengthHyper    =
    %          lengthHyperNu  =
    %          lengthScale    =
    %          magnSigma2     =


    end

end


