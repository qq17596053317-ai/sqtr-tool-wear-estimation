paperRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
clearvars -except paperRoot;
clc;

%% Reviewer-requested post-hoc sensitivity analysis
% Each NASA nominal condition is the combination of material, feed and DOC.
% Both repeated cases from a condition are held out together. Hyperparameters
% and the gate threshold are selected by inner leave-one-condition-out
% validation using only the seven remaining conditions.

addpath(paperRoot);
resultDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(resultDir, 'dir'); mkdir(resultDir); end

S = load(fullfile(paperRoot, 'mill', 'results', 'quality_aware_with_time.mat'));
T = S.timeAwareTable;

y = T.VB;
caseID = T.CaseID;
runID = T.RunID;
originalIndex = T.OriginalIndex;
elapsedTime = T.ElapsedTime;
qualityRatio = T.smcDC_SaturationRatio;

[conditionID, conditionKey] = findgroups(T(:, {'Material','Feed','DOC'}));
conditionList = unique(conditionID, 'stable');
nConditions = numel(conditionList);
assert(nConditions == 8, 'Expected eight NASA nominal conditions.');

allVariableNames = string(T.Properties.VariableNames);
sensorVariableNames = allVariableNames(8:85);
smcDCMask = startsWith(sensorVariableNames, 'smcDC_');
otherSensorNames = sensorVariableNames(~smcDCMask);

XProcess = T{:, {'DOC','Feed','Material'}};
XAllSensors = T{:, cellstr(sensorVariableNames)};
XOtherSensors = T{:, cellstr(otherSensorNames)};
XAllTime = [XProcess, XAllSensors, elapsedTime, elapsedTime.^2];
XReducedTime = [XProcess, XOtherSensors, elapsedTime, elapsedTime.^2];

lambdaGrid = logspace(-4, 4, 17);
gateGrid = [0.01 0.10 0.50];

predAll = nan(height(T), 1);
predReduced = nan(height(T), 1);
predSQTR = nan(height(T), 1);
lambdaAll = nan(nConditions, 1);
lambdaReduced = nan(nConditions, 1);
gate = nan(nConditions, 1);
gateValidationMAE = nan(nConditions, numel(gateGrid));

for k = 1:nConditions
    testCondition = conditionList(k);
    testMask = conditionID == testCondition;
    trainMask = ~testMask;

    [predAll(testMask), predReduced(testMask), predSQTR(testMask), ...
        lambdaAll(k), lambdaReduced(k), gate(k), gateValidationMAE(k,:)] = ...
        groupedGatedRidgePrediction(XAllTime, XReducedTime, qualityRatio, ...
        y, conditionID, trainMask, testMask, lambdaGrid, gateGrid);
end

assert(all(isfinite([predAll; predReduced; predSQTR])), ...
    'Non-finite leave-one-condition-out predictions.');

modelNames = ["FixedAllSensorTime"; "FixedReducedTime"; "SQTR"];
P = [predAll, predReduced, predSQTR];
nModels = size(P,2);

MAE = zeros(nModels,1);
RMSE = zeros(nModels,1);
R2 = zeros(nModels,1);
P95AbsoluteError = zeros(nModels,1);
MaximumAbsoluteError = zeros(nModels,1);
for m = 1:nModels
    e = P(:,m) - y;
    ae = abs(e);
    MAE(m) = mean(ae);
    RMSE(m) = sqrt(mean(e.^2));
    R2(m) = 1 - sum(e.^2) / sum((y - mean(y)).^2);
    P95AbsoluteError(m) = prctile(ae, 95);
    MaximumAbsoluteError(m) = max(ae);
end

conditionN = zeros(nConditions,1);
conditionCaseN = zeros(nConditions,1);
conditionMAE = zeros(nConditions,nModels);
for k = 1:nConditions
    mask = conditionID == conditionList(k);
    conditionN(k) = sum(mask);
    conditionCaseN(k) = numel(unique(caseID(mask)));
    for m = 1:nModels
        conditionMAE(k,m) = mean(abs(P(mask,m) - y(mask)));
    end
end
conditionMacroMAE = mean(conditionMAE,1)';

delta = conditionMAE(:,1) - conditionMAE(:,3); % comparator minus SQTR
rng(20260813, 'twister');
B = 10000;
bootMean = nan(B,1);
for b = 1:B
    idx = randi(nConditions, nConditions, 1);
    bootMean(b) = mean(delta(idx));
end
CI = prctile(bootMean, [2.5 97.5]);
pExact = signrank(conditionMAE(:,1), conditionMAE(:,3), ...
    'tail', 'both', 'method', 'exact');

overall = table(modelNames, MAE, conditionMacroMAE, RMSE, R2, ...
    P95AbsoluteError, MaximumAbsoluteError);

conditionTable = table(conditionList, conditionKey.Material, ...
    conditionKey.Feed, conditionKey.DOC, conditionN, conditionCaseN, ...
    conditionMAE(:,1), conditionMAE(:,2), conditionMAE(:,3), delta, ...
    'VariableNames', {'ConditionID','Material','Feed','DOC','RecordN', ...
    'CaseN','FixedAllSensorTimeMAE','FixedReducedTimeMAE','SQTR_MAE', ...
    'ComparatorMinusSQTR'});

comparison = table("FixedAllSensorTime_minus_SQTR", mean(delta), ...
    CI(1), CI(2), sum(delta > 0), sum(delta < 0), sum(delta == 0), pExact, ...
    'VariableNames', {'Comparison','MeanConditionMacroDifference', ...
    'BootstrapCI_Lower','BootstrapCI_Upper','ConditionsFavourSQTR', ...
    'ConditionsFavourComparator','TiedConditions','ExactWilcoxonP'});

hyper = table(conditionList, lambdaAll, lambdaReduced, gate, ...
    'VariableNames', {'HeldOutCondition','LambdaAll','LambdaReduced','Gate'});
for j = 1:numel(gateGrid)
    hyper.(sprintf('InnerMAE_Gate%03d', round(100*gateGrid(j)))) = ...
        gateValidationMAE(:,j);
end

predictionTable = table(originalIndex, caseID, conditionID, runID, y, ...
    qualityRatio, predAll, predReduced, predSQTR, ...
    'VariableNames', {'OriginalIndex','CaseID','ConditionID','RunID', ...
    'ActualVB','QualityRatio','PredFixedAllSensorTime', ...
    'PredFixedReducedTime','PredSQTR'});

writetable(overall, fullfile(resultDir, 'leave_one_condition_out_overall.csv'));
writetable(conditionTable, fullfile(resultDir, 'leave_one_condition_out_conditions.csv'));
writetable(comparison, fullfile(resultDir, 'leave_one_condition_out_comparison.csv'));
writetable(hyper, fullfile(resultDir, 'leave_one_condition_out_hyperparameters.csv'));
writetable(predictionTable, fullfile(resultDir, 'leave_one_condition_out_predictions.csv'));
save(fullfile(resultDir, 'leave_one_condition_out_results.mat'));

disp(overall);
disp(comparison);

