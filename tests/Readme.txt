This directory contains unit test for GPstuff. The tests can be run using
the built-in unit test framewoork (Matlab 2013b or greater) or using the
xUnit package by Steve Eddins.

Test structure in every test is basically the same: Run demo, save values
from demo, compare those saved test values to previously saved "correct"
values. Users can run these tests to determine if some calculations within
demos don't give correct answers.

Tests are named test_<demo_name>.m (e.g. test_binomial.m is a corresponding
test for demo_binomial1.m) and they can include multiple test cases each.
Individual tests can be run with command "runtests <test_file_name>".
Function runtestset can be used to run all the tests or a predetermined set
of tests.

When a test is run, the results of the demo are saved into the folder
testValues. The saved components include figures, selected workspace
variables and the log file of the command line output. Folder realValues
contain similar precomputed files for each demo.

For more information to the MATLAB unit testing framework, see Matlab
documentation.

For more information to the xUnit Test Framework package, visit 
http://www.mathworks.com/matlabcentral/fx_files/22846/11/content/matlab_xunit/doc/xunit_product_page.html

Real values from revision 990.
