function run_mylab_validation(varargin)
%RUN_MYLAB_VALIDATION  Run mylab's MATLAB-behaviour checks in MATLAB and save the results.
%
%   ON HOLD: do not run. MathWorks' licence terms forbid using MATLAB to test
%   software like mylab. See validation/README.md and
%   docs/MATLAB-TERMS-REVIEW.md in the mylab repository.
%
%   run_mylab_validation
%   run_mylab_validation('Output', 'my_results.json')
%   run_mylab_validation('MaxMinutes', 12)
%   run_mylab_validation('Only', '^lang/')      % only checks whose id matches a pattern
%
%   Runs every check in checks.json (the original and the held-out sets) and
%   evaluates every expression in exprs.json, then writes one JSON file
%   (mylab_validation_results.json in the current folder by default).
%
%   Progress is saved as it goes. If the run stops, run the same command
%   again: checks that already have a result are skipped. 'MaxMinutes' ends
%   a run cleanly after that many minutes.
%
%   Each check runs in its own temporary folder, which is deleted afterwards.
%   The kit reads only its own files and writes only the results file.

p = inputParser;
p.addParameter('Output', fullfile(pwd, 'mylab_validation_results.json'));
p.addParameter('MaxMinutes', 12);
p.addParameter('Only', '');
p.parse(varargin{:});
only = char(p.Results.Only);
outFile = char(p.Results.Output);
deadline = tic;
limit = 60 * p.Results.MaxMinutes;

kitDir = fileparts(mfilename('fullpath'));
addpath(kitDir);
checks = jsondecode(fileread(fullfile(kitDir, 'checks.json')));
exprs = jsondecode(fileread(fullfile(kitDir, 'exprs.json')));
if ~iscell(exprs)
    exprs = num2cell(exprs);
end

res = struct('matlab_version', version, 'release', version('-release'), ...
    'computer', computer, 'started', datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
    'checks', struct('set', {}, 'id', {}, 'ok', {}, 'identifier', {}, 'message', {}, 'seconds', {}), ...
    'exprs', struct('index', {}, 'expr', {}, 'value', {}));
if isfile(outFile)
    old = jsondecode(fileread(outFile));
    if isfield(old, 'checks') && ~isempty(old.checks)
        res.checks = reshape(old.checks, 1, []);
    end
    if isfield(old, 'exprs') && ~isempty(old.exprs)
        res.exprs = reshape(old.exprs, 1, []);
    end
    fprintf('Continuing: %d checks and %d expressions already done.\n', numel(res.checks), numel(res.exprs));
end
n = numel(checks);
keys = cell(1, n);
for k = 1:n
    keys{k} = [checks(k).set '|' checks(k).id];
end
oldKeys = cell(1, numel(res.checks));
for k = 1:numel(res.checks)
    oldKeys{k} = [res.checks(k).set '|' res.checks(k).id];
end
done = ismember(keys, oldKeys);

root = tempname;
mkdir(root);
cleanupRoot = onCleanup(@() rmdir(root, 's'));
startPath = path;
startDir = pwd;
startWarn = warning;

lastSave = tic;
for k = 1:n
    c = checks(k);
    if done(k) || (~isempty(only) && isempty(regexp(c.id, only, 'once')))
        continue
    end
    if toc(deadline) > limit
        save_results(outFile, res);
        fprintf('Stopped after %.0f minutes with %d of %d checks done. Run the same command again to continue.\n', ...
            toc(deadline) / 60, numel(res.checks), n);
        return
    end
    r = run_one(c, root);
    res.checks(end + 1) = r;
    done(k) = true;
    path(startPath);
    cd(startDir);
    warning(startWarn);
    if toc(lastSave) > 20
        save_results(outFile, res);
        lastSave = tic;
        fprintf('%d of %d checks done.\n', numel(res.checks), n);
    end
end

if numel(res.exprs) < numel(exprs)
    for k = numel(res.exprs) + 1:numel(exprs)
        res.exprs(end + 1) = struct('index', k, 'expr', exprs{k}, 'value', mlv_eval(exprs{k}));
    end
end
res.finished = datestr(now, 'yyyy-mm-dd HH:MM:SS');
save_results(outFile, res);
nok = sum([res.checks.ok]);
fprintf('Finished: %d of %d checks passed, %d expressions evaluated.\n', nok, numel(res.checks), numel(res.exprs));
fprintf('Results: %s\n', outFile);
end


function r = run_one(c, root)
d = tempname(root);
mkdir(d);
files = c.files;
names = {};
for j = 1:numel(files)
    f = files(j);
    full = fullfile(d, strrep(f.path, '/', filesep));
    parent = fileparts(full);
    if ~isfolder(parent)
        mkdir(parent);
    end
    fid = fopen(full, 'w', 'n', 'UTF-8');
    fwrite(fid, unicode2native(f.text, 'UTF-8'));
    fclose(fid);
    [~, nm] = fileparts(f.path);
    names{end + 1} = nm; %#ok<AGROW>
end
fid = fopen(fullfile(d, 'probe_main.m'), 'w', 'n', 'UTF-8');
fwrite(fid, unicode2native(c.code, 'UTF-8'));
fclose(fid);
cd(d);
rng('default');
format short
format loose
t = tic;
ok = true;
ident = '';
msg = '';
try
    run_isolated();
catch err
    ok = false;
    ident = err.identifier;
    msg = err.message;
end
secs = toc(t);
close all force
fclose('all');
cd(root);
for j = 1:numel(names)
    clear(names{j});
end
clear('probe_main');
clear functions %#ok<CLFUNC>
clear global
try
    rmdir(d, 's');
catch
end
r = struct('set', c.set, 'id', c.id, 'ok', ok, 'identifier', ident, 'message', msg, 'seconds', secs);
end


function run_isolated()
% The check runs in this function's own workspace, so it cannot disturb the runner.
evalc('probe_main');
end


function save_results(outFile, res)
fid = fopen(outFile, 'w', 'n', 'UTF-8');
fwrite(fid, unicode2native(jsonencode(res), 'UTF-8'));
fclose(fid);
end
