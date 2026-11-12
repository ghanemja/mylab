function cf_run_case(cf__file, cf__out)
% Run one conformance case inside this function's workspace and write every
% variable it leaves behind to cf__out as one JSON object.
% All locals here start with cf__ so they can be told apart from case variables.
% Case variables named tmp* (and ans) are not written.
cf__code = fileread(cf__file);
cf__failed = 0;
cf__msg = '';
try
    eval(cf__code);
catch cf__e
    cf__failed = 1;
    cf__msg = cf__e.message;
end
cf__names = who;
cf__fid = fopen(cf__out, 'w');
fprintf(cf__fid, '{"error":%d,"message":%s,"vars":{', cf__failed, cf_dump_value(cf__msg));
cf__first = 1;
for cf__i = 1:numel(cf__names)
    cf__n = cf__names{cf__i};
    if strncmp(cf__n, 'cf__', 4) || strncmp(cf__n, 'tmp', 3) || strcmp(cf__n, 'ans')
        continue;
    end
    try
        cf__s = cf_dump_value(eval(cf__n));
    catch cf__e2
        cf__s = '{"class":"<dump failed>","size":[0,0]}';
    end
    if cf__first
        cf__first = 0;
    else
        fprintf(cf__fid, ',');
    end
    fprintf(cf__fid, '\n"%s":%s', cf__n, cf__s);
end
fprintf(cf__fid, '\n}}\n');
fclose(cf__fid);
end
