clear; clc;

paperRoot = fileparts(mfilename('fullpath'));
dataDir = paperRoot;
files = { ...
    fullfile(dataDir, 'nuaa_orthogonal_bundle_high_resolution.csv'), ...
    fullfile(dataDir, 'phm2010_bundle_high_resolution.csv')};

for f = 1:numel(files)
    fprintf('\n===== %s =====\n', files{f});
    T = readtable(files{f}, 'VariableNamingRule', 'preserve');
    tag = string(T.experiment_tag);
    tags = unique(tag, 'stable');
    rows = table('Size', [numel(tags), 8], ...
        'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
        'VariableNames', {'Experiment','Samples','TimeMin','TimeMax','WearMin','WearMax','UniqueWear','SpearmanTimeWear'});
    for i = 1:numel(tags)
        idx = tag == tags(i);
        ti = T.timestamp(idx);
        yi = T.tool_wear(idx);
        good = isfinite(ti) & isfinite(yi);
        ti = ti(good); yi = yi(good);
        rows.Experiment(i) = tags(i);
        rows.Samples(i) = numel(ti);
        rows.TimeMin(i) = min(ti);
        rows.TimeMax(i) = max(ti);
        rows.WearMin(i) = min(yi);
        rows.WearMax(i) = max(yi);
        rows.UniqueWear(i) = numel(unique(yi));
        if numel(unique(ti)) > 1 && numel(unique(yi)) > 1
            rows.SpearmanTimeWear(i) = corr(ti, yi, 'Type', 'Spearman', 'Rows', 'complete');
        else
            rows.SpearmanTimeWear(i) = NaN;
        end
    end
    disp(rows);
end
