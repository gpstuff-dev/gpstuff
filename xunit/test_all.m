% Remove previous test values and create empty folder for new test values.
path = which('test_all.m');
path = strrep(path, 'test_all.m', 'testValues');
if exist(path,'dir') == 7
    rmdir(path, 's');
end
mkdir(path);

% Create TestSuite object suite which includes all tests from root
% directory and logger object which keeps track of the number of tests, failures and
% errors. Run tests and print logger information. Can be run from any
% directory.

path2 = cd;
cd(strrep(path,'testValues', ''));
suite = TestSuite.fromPwd();

logger = TestRunLogger;
suite.run(logger);
cd(path2);

logger
