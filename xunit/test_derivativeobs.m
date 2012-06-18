function test_suite = test_derivativeobs

%   Run specific demo and save values for comparison.
%
%   See also
%     TEST_ALL, DEMO_DERIVATIVEOBS

% Copyright (c) 2011-2012 Ville Tolvanen

initTestSuite;


  function testDemo
    % Set random number stream so test failing isn't because randomness. Run
    % demo & save test values.
    stream0 = RandStream('mt19937ar','Seed',0);
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      prevstream = RandStream.setDefaultStream(stream0);
    else
      prevstream = RandStream.setGlobalStream(stream0);
    end
    
    disp('Running: demo_derivativeobs')
    demo_derivativeobs
    path = which('test_derivativeobs.m');
    path = strrep(path,'test_derivativeobs.m', 'testValues');
    if ~(exist(path, 'dir') == 7)
      mkdir(path)
    end
    path = strcat(path, '/testDerivativeobs');
    save(path);
    
    % Set back initial random stream
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      RandStream.setDefaultStream(prevstream);
    else
      RandStream.setGlobalStream(prevstream);
    end
    drawnow;clear;close all

  % Compare test values to real values.
        
  function testPrediction
    values.real = load('realValuesDerivativeobs.mat','Eft','Varft');
    values.test = load(strrep(which('test_derivativeobs.m'), 'test_derivativeobs.m', 'testValues/testDerivativeobs'),'Eft','Varft');
    assertElementsAlmostEqual(mean(values.real.Eft),mean(values.test.Eft),'relative', 0.1);
    assertElementsAlmostEqual(mean(values.real.Varft),mean(values.test.Varft),'relative', 0.1);
    
        
  function testPredictionDerivative
    values.real = load('realValuesDerivativeobs.mat','Eft2','Varft2');
    values.test = load(strrep(which('test_derivativeobs.m'), 'test_derivativeobs.m', 'testValues/testDerivativeobs.mat'),'Eft2','Varft2');
    assertElementsAlmostEqual(mean(values.real.Eft2),mean(values.test.Eft2),'relative', 0.1);
    assertElementsAlmostEqual(mean(values.real.Varft2),mean(values.test.Varft2),'relative', 0.1);
    

