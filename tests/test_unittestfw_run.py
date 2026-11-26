# ruff: noqa: E501  (the MATLAB programs below are kept as written, one statement group per line)
"""The unit-testing framework: function-, class- and script-based tests, qualifications, constraints, fixtures, suites and runners.

Each case below is a small multi-file MATLAB program: the lines after ``@file name`` are written to that
file in a temporary folder and the lines after ``@main`` are run there as a script. A case passes when
the script reaches its end without an error; the expected values in the asserts were worked out by hand
from MATLAB's documented behaviour.
"""

import os

import pytest

from mylab import Interpreter

CASES = r"""
### ut/functions_basic
@file tAdd.m
function tests = tAdd
tests = functiontests(localfunctions);
end
function testSum(testCase)
verifyEqual(testCase, 1 + 1, 2);
end
function testTol(testCase)
verifyEqual(testCase, 0.1 + 0.2, 0.3, 'AbsTol', 1e-12);
end
@main
r = runtests('tAdd'); assert(numel(r) == 2 && all([r.Passed]));
assert(isequal({r.Name}, {'tAdd/testSum', 'tAdd/testTol'})); assert(~any([r.Failed]) && ~any([r.Incomplete])); assert(all([r.Duration] >= 0));
s = evalc('r2 = runtests(''tAdd'');'); assert(contains(s, 'Running tAdd') && contains(s, 'Done tAdd') && contains(s, '..'));
t = table(r); assert(istable(t) && height(t) == 2 && all(t.Passed) && isequal(t.Properties.VariableNames, {'Name', 'Passed', 'Failed', 'Incomplete', 'Duration', 'Details'}));
d = evalc('disp(r)'); assert(contains(d, '2 Passed, 0 Failed, 0 Incomplete'));

### ut/functions_fixtures
@file tFix.m
function tests = tFix
tests = functiontests(localfunctions);
end
function setupOnce(testCase)
testCase.TestData.base = 10; global LOG; LOG = {'setupOnce'};
end
function teardownOnce(testCase)
global LOG; LOG{end+1} = 'teardownOnce';
end
function setup(testCase)
testCase.TestData.x = testCase.TestData.base + 1; global LOG; LOG{end+1} = 'setup';
end
function teardown(testCase)
global LOG; LOG{end+1} = 'teardown';
end
function testA(testCase)
global LOG; LOG{end+1} = 'A'; verifyEqual(testCase, testCase.TestData.x, 11); testCase.TestData.x = 99;
addTeardown(testCase, @note, 'tdA1'); addTeardown(testCase, @note, 'tdA2');
end
function testB(testCase)
global LOG; LOG{end+1} = 'B'; testCase.verifyEqual(testCase.TestData.x, 11);
end
function note(s)
global LOG; LOG{end+1} = s;
end
function helperFn(x)
end
@main
global LOG
r = runtests('tFix'); assert(numel(r) == 2 && all([r.Passed]));
assert(isequal(LOG, {'setupOnce', 'setup', 'A', 'tdA2', 'tdA1', 'teardown', 'setup', 'B', 'teardown', 'teardownOnce'}));

### ut/qualification_kinds
@file tKinds.m
function tests = tKinds
tests = functiontests(localfunctions);
end
function testVerifyContinues(tc)
global HITS; verifyEqual(tc, 1, 2); HITS{end+1} = 'afterVerify'; verifyTrue(tc, true);
end
function testAssertStops(tc)
global HITS; assertEqual(tc, 1, 2); HITS{end+1} = 'afterAssert';
end
function testAssumeFilters(tc)
global HITS; assumeTrue(tc, false); HITS{end+1} = 'afterAssume';
end
function testError(tc)
error('my:id', 'boom');
end
function testPass(tc)
assertTrue(tc, true); assumeEqual(tc, 1, 1); fatalAssertTrue(tc, true);
end
function testFailNow(tc)
verifyFail(tc, 'on purpose');
end
@main
global HITS; HITS = {};
s = evalc('r = runtests(''tKinds'');'); assert(numel(r) == 6);
n = {r.Name}; k = @(name) r(strcmp(n, ['tKinds/' name]));
v = k('testVerifyContinues'); assert(v.Failed && ~v.Incomplete && ~v.Passed);
a = k('testAssertStops'); assert(a.Failed && a.Incomplete); f = k('testAssumeFilters'); assert(~f.Failed && f.Incomplete && ~f.Passed);
e = k('testError'); assert(e.Failed && e.Incomplete); p = k('testPass'); assert(p.Passed); assert(k('testFailNow').Failed);
assert(isequal(HITS, {'afterVerify'}));
assert(contains(s, 'Verification failed in tKinds/testVerifyContinues')); assert(contains(s, 'Assertion failed in tKinds/testAssertStops'));
assert(contains(s, 'Actual Value') && contains(s, 'Expected Value')); assert(contains(s, 'my:id') && contains(s, 'boom')); assert(contains(s, 'on purpose'));
assert(contains(s, 'Failure Summary')); assert(contains(s, 'tKinds/testAssumeFilters was filtered') || contains(s, 'Filtered'));
assert(~isempty(v.Details.DiagnosticRecord));
t = false; try, assertSuccess(r); catch, t = true; end; assert(t);

### ut/fatal
@file tFatal.m
function tests = tFatal
tests = functiontests(localfunctions);
end
function testA(tc)
fatalAssertTrue(tc, false);
end
function testB(tc)
end
@main
t = false; try, evalc('runtests(''tFatal'')'); catch e, t = contains(e.message, 'atal'); end; assert(t);

### ut/all_qualifications
@file tAll.m
function tests = tAll
tests = functiontests(localfunctions);
end
function testPasses(tc)
verifyEqual(tc, [1 2 3], [1 2 3]); verifyEqual(tc, {1, 'a'}, {1, 'a'}); verifyEqual(tc, struct('a', 1), struct('a', 1));
verifyEqual(tc, 1.001, 1, 'RelTol', 0.01); verifyEqual(tc, [1 NaN], [1 NaN]); verifyEqual(tc, "s", "s", 'a message');
verifyNotEqual(tc, 1, 2); verifyTrue(tc, 1 == 1); verifyFalse(tc, 1 == 2); verifyEmpty(tc, []); verifyNotEmpty(tc, 1);
verifySize(tc, ones(2, 3), [2 3]); verifyLength(tc, ones(2, 5), 5); verifyNumElements(tc, ones(2, 5), 10);
verifyClass(tc, 5, 'double'); verifyClass(tc, int8(1), ?int8); verifyInstanceOf(tc, 5, 'numeric'); verifySubstring(tc, 'hello world', 'lo w');
verifyMatches(tc, 'abc123', '[a-c]+\d+'); verifyGreaterThan(tc, 3, 2); verifyGreaterThanOrEqual(tc, 3, 3); verifyLessThan(tc, 1, 2); verifyLessThanOrEqual(tc, 2, 2);
verifyError(tc, @() error('a:b', 'x'), 'a:b'); verifyWarning(tc, @() warning('w:id', 'careful'), 'w:id'); verifyWarningFree(tc, @() 1 + 1);
verifyReturnsTrue(tc, @() true); x = verifyWarningFree(tc, @() 7); verifyEqual(tc, x, 7);
h = containers.Map(); verifySameHandle(tc, h, h); verifyNotSameHandle(tc, h, containers.Map());
end
function testEachFails(tc)
global NF
checks = {@() verifyEqual(tc, 1, int8(1)), @() verifyEqual(tc, [1 2], [1 2 3]), @() verifyEqual(tc, 1, 1.1, 'AbsTol', 0.01), ...
    @() verifyEqual(tc, {1, 2}, {1, 3}), @() verifyEqual(tc, struct('a', 1), struct('b', 1)), @() verifyNotEqual(tc, 1, 1), ...
    @() verifyTrue(tc, 1), @() verifyFalse(tc, true), @() verifyEmpty(tc, 1), @() verifyNotEmpty(tc, []), @() verifySize(tc, 1, [2 2]), ...
    @() verifyLength(tc, [1 2], 3), @() verifyNumElements(tc, 1, 2), @() verifyClass(tc, 1, 'single'), @() verifyInstanceOf(tc, 1, 'char'), ...
    @() verifySubstring(tc, 'abc', 'x'), @() verifyMatches(tc, 'abc', '^\d+$'), @() verifyGreaterThan(tc, 1, 2), @() verifyLessThan(tc, 2, 1), ...
    @() verifyError(tc, @() 1, 'a:b'), @() verifyError(tc, @() error('c:d', 'x'), 'a:b'), @() verifyWarning(tc, @() 1, 'w:id'), ...
    @() verifyWarningFree(tc, @() warning('w:id', 'm')), @() verifyReturnsTrue(tc, @() 1), @() verifyEqual(tc, 'abc', 'abd')};
NF = 0;
for k = 1:numel(checks)
    before = numel(tc.TestData);
    checks{k}(); NF = NF + 1;
end
end
@main
global NF
s = evalc('r = runtests(''tAll'');'); assert(r(strcmp({r.Name}, 'tAll/testPasses')).Passed, s); assert(NF == 25);
bad = r(strcmp({r.Name}, 'tAll/testEachFails')); assert(bad.Failed && ~bad.Incomplete); assert(numel(bad.Details.DiagnosticRecord) == 25);
assert(contains(s, 'Classes do not match') && contains(s, 'Sizes do not match') && contains(s, 'AbsoluteTolerance'));

### ut/class_based
@file MyT.m
classdef MyT < matlab.unittest.TestCase
    properties
        base
        log = {}
    end
    properties (TestParameter)
        n = {1, 2, 3}
        kind = struct('small', 1, 'big', 100)
    end
    methods (TestClassSetup)
        function classSetup(tc)
            tc.base = 5; MyT.note('classSetup');
        end
    end
    methods (TestClassTeardown)
        function classTeardown(tc), MyT.note('classTeardown'); end
    end
    methods (TestMethodSetup)
        function msetup(tc), tc.log{end+1} = 's'; MyT.note('setup'); end
    end
    methods (TestMethodTeardown)
        function mteardown(tc), MyT.note('teardown'); end
    end
    methods (Test)
        function a(tc)
            tc.verifyEqual(2 * 2, 4); tc.verifyEqual(tc.base, 5); tc.verifyEqual(tc.log, {'s'}); tc.base = 77;
            tc.addTeardown(@() MyT.note('tdA'));
        end
        function b(tc)
            tc.verifyTrue(true); tc.assertEqual([1 2], [1 2]); tc.verifyEqual(tc.base, 5);
        end
        function p(tc, n)
            tc.verifyGreaterThan(n, 0);
        end
        function q(tc, n, kind)
            tc.verifyLessThan(n, kind + 5);
        end
    end
    methods (Test, TestTags = {'Slow', 'Net'})
        function tagged(tc), tc.verifyTrue(true); end
    end
    methods (Test, ParameterCombination = 'sequential')
        function seq(tc, n), tc.verifyNotEmpty(n); end
    end
    methods (Static)
        function note(s)
            global LOG; LOG{end+1} = s;
        end
    end
    methods
        function helper(tc), end
    end
end
@main
global LOG; LOG = {};
s = evalc('r = runtests(''MyT'');'); assert(numel(r) == 2 + 3 + 6 + 1 + 3, s); assert(all([r.Passed]), s);
names = {r.Name}; assert(any(strcmp(names, 'MyT/a')) && any(strcmp(names, 'MyT/p(n=1)')) && any(strcmp(names, 'MyT/q(n=3,kind=big)')));
assert(strcmp(LOG{1}, 'classSetup') && strcmp(LOG{end}, 'classTeardown')); assert(isequal(LOG(2:5), {'setup', 'tdA', 'teardown', 'setup'}));
assert(contains(s, 'Running MyT') && contains(s, 'Done MyT'));
suite = testsuite('MyT'); assert(numel(suite) == 15); assert(strcmp(class(suite), 'matlab.unittest.Test'));
tagged = suite.selectIf('Tag', 'Slow'); assert(numel(tagged) == 1 && strcmp(tagged.Name, 'MyT/tagged') && isequal(tagged.Tags, {'Slow', 'Net'}));
r2 = runtests('MyT', 'Name', 'MyT/a'); assert(numel(r2) == 1 && strcmp(r2.Name, 'MyT/a'));
r3 = runtests('MyT', 'ProcedureName', 'p'); assert(numel(r3) == 3); r4 = runtests('MyT', 'Tag', 'Net'); assert(numel(r4) == 1);
r5 = runtests('MyT/b'); assert(numel(r5) == 1 && r5.Passed); r6 = runtests('MyT', 'Name', 'MyT/p*'); assert(numel(r6) == 3);
r7 = runtests('MyT', 'ParameterName', 'big'); assert(numel(r7) == 3);
import matlab.unittest.TestSuite
import matlab.unittest.TestRunner
s1 = TestSuite.fromClass(?MyT); assert(numel(s1) == 15); s2 = TestSuite.fromMethod(?MyT, 'b'); assert(numel(s2) == 1);
s3 = TestSuite.fromFile('MyT.m'); assert(numel(s3) == 15); s4 = TestSuite.fromFolder(pwd); assert(numel(s4) == 15);
runner = TestRunner.withTextOutput; out = evalc('res = runner.run(s2);'); assert(res.Passed && contains(out, 'Running MyT'));
quiet = TestRunner.withNoPlugins; out = evalc('res = run(quiet, s1);'); assert(isempty(out) && numel(res) == 15);
res = run(s2); assert(res.Passed); both = [s2, s1(1:2)]; assert(numel(both) == 3);
import matlab.unittest.selectors.HasName
import matlab.unittest.selectors.HasTag
import matlab.unittest.constraints.ContainsSubstring
sel = s1.selectIf(HasName(ContainsSubstring('kind=small'))); assert(numel(sel) == 3); sel2 = s1.selectIf(~HasTag('Slow') & HasName('MyT/a')); assert(numel(sel2) == 1);
tc = MyT(); rr = run(tc, 'b'); assert(rr.Passed);

### ut/class_failures
@file BadT.m
classdef BadT < matlab.unittest.TestCase
    methods (Test)
        function fails(tc)
            tc.verifyEqual([1 2 3], [1 2 4], 'values differ');
        end
        function errors(tc)
            x = [1 2 3]; y = x(7);
        end
        function filtered(tc)
            tc.assumeFail('not on this platform');
        end
        function ok(tc)
            tc.verifyThat(5, matlab.unittest.constraints.IsGreaterThan(3));
        end
    end
end
@main
s = evalc('r = runtests(''BadT'');'); assert(isequal([r.Passed], [false false false true]), s); assert(isequal([r.Failed], [true true false false])); assert(isequal([r.Incomplete], [false true true false]));
assert(contains(s, 'values differ') && contains(s, 'Verification failed in BadT/fails') && contains(s, 'Error occurred in BadT/errors'));
assert(contains(s, 'not on this platform')); assert(contains(s, 'Failure Summary'));
s2 = evalc('runtests(''BadT'', ''OutputDetail'', 0);'); assert(isempty(strtrim(s2)) || ~contains(s2, 'Running'));

### ut/constraints
@file tCon.m
function tests = tCon
tests = functiontests(localfunctions);
end
function testConstraints(tc)
import matlab.unittest.constraints.*
verifyThat(tc, 5, IsEqualTo(5)); verifyThat(tc, 5.01, IsEqualTo(5, 'Within', AbsoluteTolerance(0.1)));
verifyThat(tc, 101, IsEqualTo(100, 'Within', RelativeTolerance(0.02))); verifyThat(tc, 5.5, IsEqualTo(5, 'Within', AbsoluteTolerance(0.1) | RelativeTolerance(0.2)));
verifyThat(tc, 'ABC', IsEqualTo('abc', 'IgnoringCase', true)); verifyThat(tc, true, IsTrue); verifyThat(tc, false, IsFalse); verifyThat(tc, [], IsEmpty);
verifyThat(tc, 5, ~IsEmpty); verifyThat(tc, 5, IsGreaterThan(4) & IsLessThan(6)); verifyThat(tc, 5, IsGreaterThanOrEqualTo(5) | IsLessThanOrEqualTo(1));
verifyThat(tc, ones(2, 3), HasSize([2 3])); verifyThat(tc, 1:4, HasLength(4)); verifyThat(tc, ones(2), HasElementCount(4));
verifyThat(tc, 5, IsOfClass('double')); verifyThat(tc, 5, IsInstanceOf('numeric')); verifyThat(tc, 'hello', ContainsSubstring('ell'));
verifyThat(tc, 'hello', StartsWithSubstring('he')); verifyThat(tc, 'hello', EndsWithSubstring('lo')); verifyThat(tc, 'abc1', Matches('\d'));
verifyThat(tc, 5, IsReal); verifyThat(tc, 5, IsFinite); verifyThat(tc, [1 NaN], HasNaN); verifyThat(tc, [1 Inf], HasInf); verifyThat(tc, 5, IsScalar);
verifyThat(tc, struct('a', 1), HasField('a')); verifyThat(tc, [1 2], IsSubsetOf([1 2 3])); verifyThat(tc, [1 2 3], IsSupersetOf([1 2])); verifyThat(tc, [2 1], IsSameSetAs([1 2]));
verifyThat(tc, @() error('x:y', 'm'), Throws('x:y')); verifyThat(tc, @() 1, IssuesNoWarnings); verifyThat(tc, struct('a', 1, 'b', 2), IsEqualTo(struct('a', 1, 'b', 3), 'IgnoringFields', {'b'}));
assertThat(tc, 1, IsEqualTo(1)); assumeThat(tc, 1, IsEqualTo(1)); fatalAssertThat(tc, 1, IsEqualTo(1));
end
function testConstraintFails(tc)
import matlab.unittest.constraints.IsEqualTo
import matlab.unittest.constraints.IsGreaterThan
verifyThat(tc, 5, IsEqualTo(6), 'five is not six'); verifyThat(tc, 1, IsGreaterThan(3)); verifyThat(tc, 5, ~IsEqualTo(5));
end
@main
s = evalc('r = runtests(''tCon'');'); assert(r(1).Passed, s); assert(r(2).Failed && numel(r(2).Details.DiagnosticRecord) == 3); assert(contains(s, 'five is not six'));

### ut/fixtures
@file tFx.m
function tests = tFx
tests = functiontests(localfunctions);
end
function testTemp(tc)
global TMP
f = tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture); TMP = f.Folder; verifyTrue(tc, isfolder(TMP));
end
function testWorking(tc)
global WD
old = pwd; f = applyFixture(tc, matlab.unittest.fixtures.WorkingFolderFixture); WD = {old, pwd}; verifyNotEqual(tc, old, pwd);
fid = fopen('scratch.txt', 'w'); fclose(fid); verifyTrue(tc, isfile('scratch.txt'));
end
function testPath(tc)
global PTH
d = fullfile(pwd, 'extra'); tc.applyFixture(matlab.unittest.fixtures.PathFixture(d)); verifyEqual(tc, onlyOnPath(), 42); PTH = d;
end
function testLog(tc)
log(tc, 'a logged note'); log(tc, 1, 'terse note'); tc.log(matlab.unittest.Verbosity.Detailed, 'detailed note');
end
@file extra/onlyOnPath.m
function r = onlyOnPath()
r = 42;
end
@main
global TMP WD PTH
here = pwd; s = evalc('r = runtests(''tFx'');'); assert(all([r.Passed]), s); assert(~isfolder(TMP)); assert(strcmp(pwd, here)); assert(strcmp(WD{1}, here) && ~isfolder(WD{2}));
assert(~contains(path, PTH)); assert(contains(s, 'terse note') && ~contains(s, 'a logged note'));
s2 = evalc('runtests(''tFx'', ''LoggingLevel'', 3);'); assert(contains(s2, 'a logged note') && contains(s2, 'detailed note'));

### ut/script_based
@file scriptTest.m
% shared variables
base = 10;
tol = 1e-9;

%% Test 1: sum works
x = base + 1;
assert(x == 11)

%% Test failing
assert(base == 0, 'base is not zero')

%% third
y = helper(base);
assert(abs(y - 20) < tol)

function r = helper(v)
r = 2 * v;
end
@main
s = evalc('r = runtests(''scriptTest'');'); assert(numel(r) == 3, s); assert(isequal([r.Passed], [true false true]), s);
assert(isequal({r.Name}, {'scriptTest/Test1_SumWorks', 'scriptTest/TestFailing', 'scriptTest/third'})); assert(contains(s, 'base is not zero'));
assert(~exist('x', 'var') && ~exist('base', 'var'));

### ut/folder_and_suite
@file tests/tOne.m
function tests = tOne
tests = functiontests(localfunctions);
end
function testA(tc), verifyTrue(tc, true); end
@file tests/TwoTest.m
classdef TwoTest < matlab.unittest.TestCase
    methods (Test)
        function x(tc), tc.verifyEqual(1, 1); end
        function y(tc), tc.verifyEqual(1, 1); end
    end
end
@file tests/helperFn.m
function r = helperFn()
r = 1;
end
@file tests/sub/tDeep.m
function tests = tDeep
tests = functiontests(localfunctions);
end
function testD(tc), verifyTrue(tc, true); end
@file tests/+pk/PkTest.m
classdef PkTest < matlab.unittest.TestCase
    methods (Test)
        function z(tc), tc.verifyEqual(1, 1); end
    end
end
@main
s = evalc('r = runtests(''tests'');'); assert(numel(r) == 3 && all([r.Passed]), s);
r2 = runtests('tests', 'IncludeSubfolders', true); assert(numel(r2) == 4);
suite = testsuite(fullfile(pwd, 'tests')); assert(numel(suite) == 3); assert(strcmp(suite(1).BaseFolder, fullfile(pwd, 'tests')));
addpath(fullfile(pwd, 'tests')); r3 = runtests({'tOne', 'TwoTest'}); assert(numel(r3) == 3);
p = matlab.unittest.TestSuite.fromPackage('pk'); assert(numel(p) == 1 && strcmp(p.Name, 'pk.PkTest/z')); rp = run(p); assert(rp.Passed);
cd tests; s = evalc('r4 = runtests;'); cd ..; assert(numel(r4) == 3 && all([r4.Passed]));
t = false; try, runtests('helperFn'); catch, t = true; end; assert(t); t = false; try, runtests('noSuchThing123'); catch, t = true; end; assert(t);

### ut/interactive
tc = matlab.unittest.TestCase.forInteractiveUse; s = evalc('tc.verifyEqual(1, 1)'); assert(contains(s, 'passed'));
s = evalc('verifyEqual(tc, 1, 2)'); assert(contains(s, 'failed') && contains(s, 'Actual Value'));
t = false; try, evalc('tc.assertTrue(false)'); catch e, t = true; end; assert(t);
t = false; try, evalc('tc.assumeTrue(false)'); catch e, t = true; end; assert(t);

### ut/params_and_setup_failures
@file ParT.m
classdef (TestTags = {'All'}) ParT < matlab.unittest.TestCase
    properties (TestParameter)
        a = {1, 2}
        b = {10, 20}
        c = struct('x', 'X', 'y', 'Y')
    end
    methods (Test, ParameterCombination = 'sequential')
        function seq(tc, a, b), tc.verifyEqual(b, a * 10); end
    end
    methods (Test, ParameterCombination = 'pairwise')
        function pair(tc, a, b, c), tc.verifyNotEmpty(c); end
    end
    methods (Test, ParameterCombination = 'exhaustive', TestTags = {'Big'})
        function full(tc, a, b, c), tc.verifyClass(c, 'char'); end
    end
end
@file SetupFails.m
classdef SetupFails < matlab.unittest.TestCase
    methods (TestClassSetup)
        function boom(tc), error('setup:bad', 'class setup broke'); end
    end
    methods (Test)
        function one(tc), end
        function two(tc), end
    end
end
@file MethodSetupFails.m
classdef MethodSetupFails < matlab.unittest.TestCase
    properties
        ran = false
    end
    methods (TestMethodSetup)
        function s(tc), tc.assumeTrue(false, 'skip all'); end
    end
    methods (Test)
        function one(tc), tc.ran = true; end
    end
end
@main
s = testsuite('ParT'); names = {s.Name}; assert(sum(startsWith(names, 'ParT/seq')) == 2 && sum(startsWith(names, 'ParT/full')) == 8);
np = sum(startsWith(names, 'ParT/pair')); assert(np >= 4 && np < 8); assert(any(strcmp(names, 'ParT/seq(a=1,b=10)')) && any(strcmp(names, 'ParT/full(a=2,b=20,c=y)')));
r = run(s); assert(all([r.Passed])); assert(numel(s.selectIf('Tag', 'Big')) == 8 && numel(s.selectIf('Tag', 'All')) == numel(s));
ps = s.selectIf('ParameterProperty', 'c'); assert(numel(ps) == 8 + np); assert(strcmp(s(1).Parameterization(1).Property, 'a'));
out = evalc('r = runtests(''SetupFails'');'); assert(numel(r) == 2 && all([r.Failed]) && all([r.Incomplete])); assert(contains(out, 'class setup broke'));
out = evalc('r = runtests(''MethodSetupFails'');'); assert(~r.Failed && r.Incomplete && ~r.Passed); assert(contains(out, 'skip all'));

### ut/custom_fixture_and_teardown_order
@file MyFx.m
classdef MyFx < matlab.unittest.fixtures.Fixture
    properties
        tag = 'fx'
    end
    methods
        function setup(f)
            global LOG; LOG{end+1} = 'fx-setup'; f.addTeardown(@() noteIt('fx-added'));
        end
        function teardown(f)
            global LOG; LOG{end+1} = 'fx-teardown';
        end
    end
end
@file noteIt.m
function noteIt(s)
global LOG; LOG{end+1} = s;
end
@file OrdT.m
classdef OrdT < matlab.unittest.TestCase
    methods (TestClassSetup)
        function cs(tc), tc.addTeardown(@noteIt, 'class-td'); noteIt('class-setup'); end
    end
    methods (TestMethodSetup)
        function ms(tc), tc.addTeardown(@noteIt, 'setup-td'); noteIt('m-setup'); end
    end
    methods (TestMethodTeardown)
        function mt(tc), noteIt('m-teardown'); end
    end
    methods (Test)
        function t1(tc)
            f = tc.applyFixture(MyFx()); tc.verifyEqual(f.tag, 'fx'); tc.addTeardown(@noteIt, 'test-td'); noteIt('body');
        end
    end
end
@main
global LOG; LOG = {};
r = runtests('OrdT'); assert(r.Passed);
assert(isequal(LOG, {'class-setup', 'm-setup', 'fx-setup', 'body', 'test-td', 'fx-teardown', 'fx-added', 'm-teardown', 'setup-td', 'class-td'}));

### ut/junit_and_runner_options
@file tJ.m
function tests = tJ
tests = functiontests(localfunctions);
end
function testGood(tc), verifyTrue(tc, true); end
function testBad(tc), verifyEqual(tc, 1, 2); end
@main
import matlab.unittest.TestRunner
import matlab.unittest.plugins.XMLPlugin
suite = testsuite('tJ'); runner = TestRunner.withNoPlugins; runner.addPlugin(XMLPlugin.producingJUnitFormat('res.xml'));
r = runner.run(suite); assert(isequal([r.Passed], [true false]) || isequal([r.Passed], [false true]));
x = fileread('res.xml'); assert(contains(x, '<testsuite name="tJ" tests="2" failures="1"') && contains(x, 'name="testBad"') && contains(x, '<failure>'));
quietOut = evalc('r2 = run(TestRunner.withTextOutput(''OutputDetail'', 0), suite);'); assert(isempty(strtrim(quietOut)));
t = table(r); assert(sum(t.Failed) == 1 && iscell(t.Name));
t0 = false; try, runtests('tJ', 'NoSuchOption', 1); catch, t0 = true; end; assert(t0);
t1 = false; try, runtests('tJ', 'UseParallel', true); catch, t1 = true; end; assert(t1);

### ut/compare_values
@file tCmp.m
function tests = tCmp
tests = functiontests(localfunctions);
end
function testObjectsAndTables(tc)
verifyEqual(tc, VObj(3), VObj(3)); verifyNotEqual(tc, VObj(3), VObj(4));
verifyEqual(tc, containers.Map().Count, uint64(0)); verifyEqual(tc, datetime(2020, 1, 2), datetime(2020, 1, 2));
verifyEqual(tc, single(1), single(1)); verifyEqual(tc, 1 + 2i, 1 + 2i); verifyEqual(tc, {1, {2, 'x'}}, {1, {2, 'x'}});
verifyEqual(tc, [1.0 2.0], [1.05 2.05], 'AbsTol', 0.1); verifyEqual(tc, struct('a', {1, 2}), struct('a', {1, 2}));
verifyEqual(tc, {1.0001, 'k'}, {1, 'k'}, 'RelTol', 1e-3); verifyEqual(tc, [Inf -Inf NaN], [Inf -Inf NaN]); verifyEqual(tc, '', '');
verifyEqual(tc, 1000, 1001, 'AbsTol', 0.5, 'RelTol', 0.01); verifyEqual(tc, true, true); verifyEqual(tc, @sin, @sin);
end
function testMismatches(tc)
verifyNotEqual(tc, 1, single(1)); verifyNotEqual(tc, 'a', "a"); verifyNotEqual(tc, [1 2], [1; 2]); verifyNotEqual(tc, {1}, {2});
verifyNotEqual(tc, struct('a', 1), struct('a', 2)); verifyNotEqual(tc, true, 1); verifyNotEqual(tc, [1 NaN], [1 2]); verifyNotEqual(tc, Inf, -Inf);
end
@file VObj.m
classdef VObj
    properties
        v
    end
    methods
        function o = VObj(v), o.v = v; end
    end
end
@main
out = evalc('r = runtests(''tCmp'');'); assert(all([r.Passed]), out);
"""


def _parse(text):
    cases, cur = [], None
    for raw in text.split("\n"):
        if raw.startswith("### "):
            cur = {"id": raw[4:].strip(), "files": {}, "main": [], "target": "main"}
            cases.append(cur)
        elif cur is None:
            continue
        elif raw.startswith("@file "):
            cur["target"] = raw[6:].strip()
            cur["files"][cur["target"]] = []
        elif raw.strip() == "@main":
            cur["target"] = "main"
        else:
            (cur["main"] if cur["target"] == "main" else cur["files"][cur["target"]]).append(raw)
    return cases


_ALL = _parse(CASES)


@pytest.mark.parametrize("case", _ALL, ids=[c["id"] for c in _ALL])
def test_case(case, tmp_path, monkeypatch):
    for rel, body in case["files"].items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(body).strip("\n") + "\n", encoding="utf-8")
    main = tmp_path / "main_script.m"
    main.write_text("\n".join(case["main"]).strip("\n") + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = []
    interp = Interpreter(out=out.append)
    interp.run_file(os.fspath(main))
