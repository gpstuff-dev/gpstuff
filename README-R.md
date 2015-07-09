Using GPstuff from R
--------------------

GPstuff can be used from R by using RcppOctave package. RcppOctave
allows to call any Octave functions and browse their documentation
from R, and pass variables between R and Octave.

** Instructions **

Start R and install RcppOctave
```
 install.packages("RcppOctave")
 require(RcppOctave)
```

Set path to GPstuff
```
 path <- .O$genpath("/path/to/gpstuff")
 .O$addpath(path)
```

Run demo
```
 o_source("/path/to/gpstuff/gp/demo_classific.m")
```

or run Octave commands line by line
```
 o_eval("L = strrep(S, 'demo_classific.m', 'demodata/synth.tr')")
 o_eval("x = load(L)")
 o_eval("y=x(:,end);")
 o_eval("y = 2.*y-1;")
 ...
```

More examples to follow.
