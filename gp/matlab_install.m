function matlab_install(suiteSparse_path)
%  Matlab function to compile all the c-files to mex in the GPstuff toolbox.
%
%  Some of the sparse GP functionalities in the toolbox require 
%  SuiteSparse toolbox by Tim Davis. First install SuiteSparse from: 
%    http://www.cise.ufl.edu/research/sparse/SuiteSparse/current/SuiteSparse/
%
%  Note! Install also Metis 4.0.1 as mentioned under header "Other
%        packages required:".           
%
%  After this, install the GPstuff package:
%
%   Run matlab_install( suitesparse_path ) in the present directory. 
%   Here suitesparse_path is a string telling the path to SuiteSparse 
%   package (for example, '/matlab/toolbox/SuiteSparse/'). Note! It is
%   important that suitesparse_path is in right format. Include also
%   the last '/' sign in it.
% 
%   The function matlab_install compiles the mex-files and prints on
%   the screen, which directories should be added to Matlab paths. 
    
% Copyright (c) 2008-2010 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 2 or later); please refer to the file 
% License.txt, included with the software, for details.


    
    if nargin < 1
        suiteSparse_path = [];
        fprintf('\n The path to the SuiteSparse package is not provided. \n')
        fprintf('\n Installing GPstuff without compactly supported covariance functions. \n')
        fprintf(' You are not able to use the following covariance functions:  \n')
        fprintf(' gpcf_ppcs0  \n gpcf_ppcs1 \n gpcf_ppcs2 \n gpcf_ppcs3 \n\n\n')
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
    gp_install(suiteSparse_path)    
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
    
    for i = 1:length(S)
       addpath(S{i}); 
    end
   
    fprintf ('\n The following paths have been added.  You may wish to add them\n') ;
    fprintf ('permanently, using the MATLAB pathtool command or copying the below\n') ;
    fprintf ('lines to your startup.m file. \n');
    fprintf ('addpath %s\n', S{1}) ;
    fprintf ('addpath %s\n', S{2}) ;
    fprintf ('addpath %s\n', S{3}) ;
    fprintf ('addpath %s\n', S{4}) ;
    fprintf ('addpath %s\n', S{5}) ;
    fprintf ('addpath %s\n', S{6}) ;
            
end