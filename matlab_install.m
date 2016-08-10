function matlab_install(SuiteSparse_path)
%  Matlab function to compile all the c-files to mex in the GPstuff toolbox.
%
%  Some of the sparse GP functionalities in the toolbox require 
%  SuiteSparse toolbox by Tim Davis:
%    http://www.cise.ufl.edu/research/sparse/SuiteSparse/current/SuiteSparse/
%
%  This package includes the SuiteSparse version 4.4.2. 
% 
%  * To install without SuiteSparse run matlab_install
%  * To install with SuiteSparse run matlab_install('SuiteSparseOn')
%
%   The function matlab_install compiles the mex-files and prints on
%   the screen, which directories should be added to Matlab paths. 
    
% Copyright (c) 2008-2012, 2016 Jarno Vanhatalo
% Copyright (c) 2016            Eero Siivola
    
% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.


    
    if nargin < 1
        SuiteSparse_path = [];
        fprintf('\n The path to the SuiteSparse package is not provided. \n')
        fprintf('\n Installing GPstuff without compactly supported covariance functions. \n')
        fprintf(' You are not able to use the following covariance functions:  \n')
        fprintf(' gpcf_ppcs0  \n gpcf_ppcs1 \n gpcf_ppcs2 \n gpcf_ppcs3 \n\n\n')
    elseif strcmp(SuiteSparse_path, 'SuiteSparseOn')
        cdir = pwd;        
        cd SuiteSparse % Change this to correspond your own settings!
        SuiteSparse_path = path_spaces([pwd '/']);
        
        
        % Compile SuiteSparse
        fprintf('Compiling SuiteSparse. This may take a while \n \n')
        SuiteSparse_install(false);
        
        paths{1} = SuiteSparse_path;
        paths{2} = [SuiteSparse_path 'UMFPACK/MATLAB'];
        paths{3} = [SuiteSparse_path 'CHOLMOD/MATLAB'];
        paths{4} = [SuiteSparse_path 'AMD/MATLAB'];
        paths{5} = [SuiteSparse_path 'COLAMD/MATLAB'];
        paths{6} = [SuiteSparse_path 'CCOLAMD/MATLAB'];
        paths{7} = [SuiteSparse_path 'CAMD/MATLAB'];
        paths{8} = [SuiteSparse_path 'CXSparse/MATLAB/Demo'];
        paths{9} = [SuiteSparse_path 'CXSparse/MATLAB/CSparse'];
        paths{10} = [SuiteSparse_path 'LDL/MATLAB'];
        paths{11} = [SuiteSparse_path 'BTF/MATLAB'];
        paths{12} = [SuiteSparse_path 'KLU/MATLAB'];
        paths{13} = [SuiteSparse_path 'SPQR/MATLAB'];
        paths{14} = [SuiteSparse_path 'RBio/RBio'];
        paths{15} = [SuiteSparse_path 'MATLAB_Tools'];
        paths{16} = [SuiteSparse_path 'MATLAB_Tools/Factorize'];
        paths{17} = [SuiteSparse_path 'MATLAB_Tools/MESHND'];
        paths{18} = [SuiteSparse_path 'MATLAB_Tools/LINFACTOR'];
        paths{19} = [SuiteSparse_path 'MATLAB_Tools/find_components'];
        paths{20} = [SuiteSparse_path 'MATLAB_Tools/GEE'];
        paths{21} = [SuiteSparse_path 'MATLAB_Tools/shellgui'];
        paths{22} = [SuiteSparse_path 'MATLAB_Tools/waitmex'];
        paths{23} = [SuiteSparse_path 'MATLAB_Tools/spqr_rank'];
        paths{24} = [SuiteSparse_path 'MATLAB_Tools/spqr_rank/SJget'];
        paths{25} = [SuiteSparse_path 'MATLAB_Tools/UFcollection'];
        paths{26} = [SuiteSparse_path 'MATLAB_Tools/SSMULT'];
        paths{27} = [SuiteSparse_path 'MATLAB_Tools/dimacs10'];
        paths{28} = [SuiteSparse_path 'MATLAB_Tools/spok'];
        paths{29} = [SuiteSparse_path 'MATLAB_Tools/sparseinv'];
        
        cd(cdir)
        fprintf('Compiling GPstuff. This may take a while \n \n')
    else 
        error('Unknown input argument. See help matlab_install for usage.')
    end
            
    % Go to diag/ and compile the mex-functions
    fprintf('\n Compiling files in diag.\n \n')
    cd('diag')
    diag_install
    cd('..')
        
    % Go to dist/ and compile the mex-functions        
    fprintf('\n Compiling files in dist.\n \n')
    cd('dist')
    dist_install    
    cd('..')
    
    % Go to gp/ and compile the mex-functions        
    fprintf('\n Compiling files in gp.\n \n')
    cd('gp')
    gp_install(SuiteSparse_path)    
    cd('..')       
        
    % Go to mc/ and compile the mex-functions
    fprintf('\n Compiling files in mc. \n \n')
    cd('mc')
    mc_install    
    cd('..')       
    
    PP = pwd;
    S{1} = [PP '/diag']; 
    S{2} = [PP '/dist']; 
    S{3} = [PP '/gp']; 
    S{4} = [PP '/mc']; 
    S{5} = [PP '/misc']; 
    S{6} = [PP '/optim']; 
    S{7} = [PP '/tests']; 
    
    fprintf ('\n The following paths have been added.  You may wish to add them\n') ;
    fprintf ('permanently, using the MATLAB pathtool command or copying the below\n') ;
    fprintf ('lines to your startup.m file. \n\n');
    for i = 1:length(S)
       addpath(S{i}); 
       fprintf ('addpath %s\n', S{i}) ;
    end

    if nargin==1
        fprintf ('\n')
        for k = 1:length (paths)
            fprintf ('addpath %s\n', paths {k}) ;
        end
    end
end

function path = path_spaces(path)
% Build correct path if path includes spaces

space_ind=strfind(path, ' ');
path=strrep(path, ' ', ''' ');
space_ind=space_ind+length(space_ind);
for i=1:length(space_ind)
  
  indd=strfind(path(space_ind(i):end), '/');
  path=[path(1:space_ind(i)+indd(1)-2) '''/' path(space_ind(i)+indd(1):end)];
  space_ind=space_ind+1;
  
end

end
