function dist_install

    if ispc   % A windows version of Matlab
      if ~exist('OCTAVE_VERSION','builtin')
        mex -O -output cond_invgam_invgam1 winCsource\cond_invgam_invgam1.c winCsource\ars.c winCsource\rand.c
        mex -O -output digamma1 winCsource\digamma1.c
        mex -O -output dirrand winCsource\dirrand.c
        mex -O -output exprand winCsource\exprand.c
        mex -O -output gamrand winCsource\gamrand.c winCsource\rand.c
        mex -O -output gamrand1 winCsource\gamrand1.c winCsource\rand.c
        mex -O -output invgamrand winCsource\invgamrand.c winCsource\rand.c
        mex -O -output invgamrand1 winCsource\invgamrand1.c winCsource\rand.c
        mex -O -output tanh_f winCsource\tanh_f.c
        mex -O -output trand winCsource\trand.c winCsource\rand.c
      else
        mex --output cond_invgam_invgam1.mex winCsource\cond_invgam_invgam1.c winCsource\ars.c winCsource\rand.c
        mex --output digamma1.mex winCsource\digamma1.c
        mex --output dirrand.mex winCsource\dirrand.c
        mex --output exprand.mex winCsource\exprand.c
        mex --output gamrand.mex winCsource\gamrand.c winCsource\rand.c
        mex --output gamrand1.mex winCsource\gamrand1.c winCsource\rand.c
        mex --output invgamrand.mex winCsource\invgamrand.c winCsource\rand.c
        mex --output invgamrand1.mex winCsource\invgamrand1.c winCsource\rand.c
        mex --output tanh_f.mex winCsource\tanh_f.c
        mex --output trand.mex winCsource\trand.c winCsource\rand.c
      end
    else
      if ~exist('OCTAVE_VERSION','builtin')        
        mex -O -output cond_invgam_invgam1 linuxCsource/cond_invgam_invgam1.c linuxCsource/ars.c linuxCsource/rand.c;
        mex -O -output digamma1 linuxCsource/digamma1.c;
        mex -O -output dirrand linuxCsource/dirrand.c;
        mex -O -output exprand linuxCsource/exprand.c;
        mex -O -output gamrand linuxCsource/gamrand.c linuxCsource/rand.c;
        mex -O -output gamrand1 linuxCsource/gamrand1.c linuxCsource/rand.c;
        mex -O -output invgamrand linuxCsource/invgamrand.c linuxCsource/rand.c;
        mex -O -output invgamrand1 linuxCsource/invgamrand1.c linuxCsource/rand.c;
        mex -O -output tanh_f linuxCsource/tanh_f.c;
        mex -O -output trand linuxCsource/trand.c linuxCsource/rand.c;
      else
        mex --output cond_invgam_invgam1.mex linuxCsource/cond_invgam_invgam1.c linuxCsource/ars.c linuxCsource/rand.c;
        mex --output digamma1.mex linuxCsource/digamma1.c;
        mex --output dirrand.mex linuxCsource/dirrand.c;
        mex --output exprand.mex linuxCsource/exprand.c;
        mex --output gamrand.mex linuxCsource/gamrand.c linuxCsource/rand.c;
        mex --output gamrand1.mex linuxCsource/gamrand1.c linuxCsource/rand.c;
        mex --output invgamrand.mex linuxCsource/invgamrand.c linuxCsource/rand.c;
        mex --output invgamrand1.mex linuxCsource/invgamrand1.c linuxCsource/rand.c;
        mex --output tanh_f.mex linuxCsource/tanh_f.c;
        mex --output trand.mex linuxCsource/trand.c linuxCsource/rand.c;
      end
    end
    