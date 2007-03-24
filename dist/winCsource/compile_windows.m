%
% Compile the C-source functions with Matlab. 
%
% Converts *.c functions to .mex* versions of desktop interface 
% in use.
%
% 


mex -O -output bbmean bbmean.c
mex -O -output bbprctile bbprctile.c
mex -O -output catrand catrand.c binsgeq.c
mex -O -output cond_invgam_invgam1 cond_invgam_invgam1.c ars.c rand.c
mex -O -output dirrand dirrand.c
mex -O -output exprand exprand.c
mex -O -output gamrand gamrand.c rand.c
mex -O -output gamrand1 gamrand1.c rand.c
mex -O -output gpexpcov gpexpcov.c 
mex -O -output gpexptrcov gpexptrcov.c 
mex -O -output invgamrand invgamrand.c rand.c
mex -O -output invgamrand1 invgamrand1.c rand.c
mex -O -output resampres resampres.c binsgeq.c
mex -O -output resampsim resampsim.c binsgeq.c
mex -O -output resampstr resampstr.c binsgeq.c
mex -O -output resampdet resampdet.c binsgeq.c
mex -O -output tanh_f tanh_f.c
mex -O -output trand trand.c rand.c

% The following routines use LAPACK libraries. The path must specified on the 
% mex command line, depending on the compiler. 
%
% Replace the path D:\MATLAB6p5p1 by the matlab installation directory.
% Here we provide commands for the bundled compiler that comes with matlab 
% and MS Visual C++ 

Compiler='Lcc';
% or Compiler='msvc';

% Use Lcc compiler, bundled with matlab
if strcmp(Compiler,'Lcc')==1,
  mex -O -output gp2fwd gp2fwd.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output gp2fwds gp2fwds.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output gpexpedata gpexpedata.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output gpexptrcovinv gpexptrcovinv.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output gpvalues2 gpvalues.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output mlp2bkp mlp2bkp.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
  mex -O -output mlp2fwd mlp2fwd.c D:\MATLAB6p5p1\extern\lib\win32\lcc\libmwlapack.lib -f lccopts.bat
end

% Use MS Visual C++ compiler. If it is not the default compiler, add appropriate mexopts 
% file, such as -f mscv60opts.bat
if strcmp(Compiler,'msvc')==1,
  mex -O -output gp2fwd gp2fwd.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output gp2fwds gp2fwds.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output gpexpedata gpexpedata.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output gpexptrcovinv gpexptrcovinv.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output gpvalues gpvalues.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output mlp2bkp mlp2bkp.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
  mex -O -output mlp2fwd mlp2fwd.c D:\MATLAB6p5p1\extern\lib\win32\microsoft\msvc60\libmwlapack.lib
end

