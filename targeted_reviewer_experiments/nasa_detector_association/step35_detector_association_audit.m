paperRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
clearvars -except paperRoot;
clc;

%% Exploratory audit of NASA quality evidence
% This analysis quantifies association with VB and nominal condition. It
% cannot determine whether the observed upper-end plateau is a hardware fault.

resultDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(resultDir, 'dir'); mkdir(resultDir); end

S = load(fullfile(paperRoot, 'mill', 'results', 'quality_aware_with_time.mat'));
T = S.timeAwareTable;
q = T.smcDC_SaturationRatio;
y = T.VB;
caseID = T.CaseID;
[conditionID, conditionKey] = findgroups(T(:, {'Material','Feed','DOC'}));
conditionList = unique(conditionID, 'stable');

rhoOverall = corr(q, y, 'Type', 'Spearman', 'Rows', 'complete');
qWithin = q;
yWithin = y;
for c = unique(caseID, 'stable')'
    mask = caseID == c;
    qWithin(mask) = q(mask) - mean(q(mask));
    yWithin(mask) = y(mask) - mean(y(mask));
end
rhoWithinCase = corr(qWithin, yWithin, 'Type', 'Spearman', 'Rows', 'complete');

rng(20260813, 'twister');
caseList = unique(caseID, 'stable');
B = 10000;
bootOverall = nan(B,1);
bootWithin = nan(B,1);
for b = 1:B
    sampledCases = caseList(randi(numel(caseList), numel(caseList), 1));
    qb = [];
    yb = [];
    qwb = [];
    ywb = [];
    for j = 1:numel(sampledCases)
        mask = caseID == sampledCases(j);
        qj = q(mask);
        yj = y(mask);
        qb = [qb; qj]; %#ok<AGROW>
        yb = [yb; yj]; %#ok<AGROW>
        qwb = [qwb; qj - mean(qj)]; %#ok<AGROW>
        ywb = [ywb; yj - mean(yj)]; %#ok<AGROW>
    end
    bootOverall(b) = corr(qb, yb, 'Type', 'Spearman', 'Rows', 'complete');
    bootWithin(b) = corr(qwb, ywb, 'Type', 'Spearman', 'Rows', 'complete');
end

rhoOverallCI = prctile(bootOverall, [2.5 97.5]);
rhoWithinCI = prctile(bootWithin, [2.5 97.5]);

nConditions = numel(conditionList);
RecordN = zeros(nConditions,1);
CaseN = zeros(nConditions,1);
MedianQ = zeros(nConditions,1);
MaximumQ = zeros(nConditions,1);
Trigger1N = zeros(nConditions,1);
Trigger10N = zeros(nConditions,1);
Trigger50N = zeros(nConditions,1);
Trigger1Rate = zeros(nConditions,1);
MedianVB = zeros(nConditions,1);
for k = 1:nConditions
    mask = conditionID == conditionList(k);
    RecordN(k) = sum(mask);
    CaseN(k) = numel(unique(caseID(mask)));
    MedianQ(k) = median(q(mask));
    MaximumQ(k) = max(q(mask));
    Trigger1N(k) = sum(q(mask) >= 0.01);
    Trigger10N(k) = sum(q(mask) >= 0.10);
    Trigger50N(k) = sum(q(mask) >= 0.50);
    Trigger1Rate(k) = Trigger1N(k) / RecordN(k);
    MedianVB(k) = median(y(mask));
end

conditionAudit = table(conditionList, conditionKey.Material, ...
    conditionKey.Feed, conditionKey.DOC, RecordN, CaseN, MedianQ, MaximumQ, ...
    Trigger1N, Trigger10N, Trigger50N, Trigger1Rate, MedianVB, ...
    'VariableNames', {'ConditionID','Material','Feed','DOC','RecordN', ...
    'CaseN','MedianQ','MaximumQ','Trigger1N','Trigger10N','Trigger50N', ...
    'Trigger1Rate','MedianVB'});

association = table(["Overall record-level"; "Within-case centred"], ...
    [rhoOverall; rhoWithinCase], ...
    [rhoOverallCI(1); rhoWithinCI(1)], ...
    [rhoOverallCI(2); rhoWithinCI(2)], ...
    'VariableNames', {'Association','SpearmanRho','CaseClusterCI_Lower', ...
    'CaseClusterCI_Upper'});

writetable(conditionAudit, fullfile(resultDir, 'detector_condition_audit.csv'));
writetable(association, fullfile(resultDir, 'detector_vb_association.csv'));
save(fullfile(resultDir, 'detector_association_audit.mat'));

disp(conditionAudit);
disp(association);

