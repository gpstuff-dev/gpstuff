%
% Compile the C-source functions with Matlab. 
%
% Converts *.c functions to mex-files for Unix type of user interfaces.
%

mex -O -output bbmean bbmean.c
mex -O -output bbprctile bbprctile.c
mex -O -output catrand catrand.c binsgeq.c
mex -O -output cond_invgam_invgam1 cond_invgam_invgam1.c ars.c rand.c
mex -O -output digamma1 digamma1.c
mex -O -output dirrand dirrand.c
mex -O -output exprand exprand.c
mex -O -output gamrand gamrand.c rand.c
mex -O -output gamrand1 gamrand1.c rand.c
mex -O -output gp2fwd gp2fwd.c
mex -O -output gp2fwds gp2fwds.c
mex -O -output gpexpcov gpexpcov.c
mex -O -output gpexpedata gpexpedata.c
mex -O -output gpexptrcov gpexptrcov.c
mex -O -output gpexptrcovinv gpexptrcovinv.c
mex -O -output gpvalues gpvalues.c 
mex -O -output invgamrand invgamrand.c rand.c
mex -O -output invgamrand1 invgamrand1.c rand.c
mex -O -output mlp2bkp mlp2bkp.c
mex -O -output mlp2fwd mlp2fwd.c
mex -O -output resampres resampres.c binsgeq.c
mex -O -output resampsim resampsim.c binsgeq.c
mex -O -output resampstr resampstr.c binsgeq.c
mex -O -output resampdet resampdet.c binsgeq.c
mex -O -output tanh_f tanh_f.c
mex -O -output trand trand.c rand.c
