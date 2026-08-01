clear;
clc;
close all;

%% 1. Scope and prespecified operating points
% This analysis does not change the SQTR point prediction. It calibrates a
% separate alert threshold from outer-training cases only, using inner
% leave-one-case-out predictions. The primary operating point is the
% VB >= 0.40 mm alert with a target inner sensitivity of 0.90. Sensitivity
% targets 0.85 and 0.95 and other wear thresholds are reported as an
% operating-characteristic analysis rather than alternative primary tests.

scriptFolder = fileparts(mfilename("fullpath"));
projectFolder = fileparts(fileparts(scriptFolder));
addpath(projectFolder);

resultFolder = fullfile(projectFolder,"mill","results");
outputFolder = scriptFolder;
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

timeData = load(fullfile(resultFolder,"quality_aware_with_time.mat"));
modelData = load(fullfile(resultFolder,"time_model_results.mat"));
T = timeData.timeAwareTable;

y = T.VB;
caseID = T.CaseID;
caseList = unique(caseID);
numberOfCases = numel(caseList);
numberOfSamples = height(T);
elapsedTime = T.ElapsedTime;
saturationRatio = T.smcDC_SaturationRatio;

allVariableNames = string(T.Properties.VariableNames);
sensorVariableNames = allVariableNames(8:85);
smcDCMask = startsWith(sensorVariableNames,"smcDC_");
otherSensorNames = sensorVariableNames(~smcDCMask);
XProcess = T{:,{'DOC','Feed','Material'}};
XAllSensors = T{:,cellstr(sensorVariableNames)};
XOtherSensors = T{:,cellstr(otherSensorNames)};
XAllTime = [XProcess,XAllSensors,elapsedTime,elapsedTime.^2];
XExcludedTime = [XProcess,XOtherSensors,elapsedTime,elapsedTime.^2];

primaryPrediction = modelData.predictionTable.PredTimeGated;
hyperparameters = modelData.thresholdSelectionTable;

wearThresholds = [0.30 0.40 0.60 0.80];
targetSensitivityGrid = [0.85 0.90 0.95];
primaryWearThreshold = 0.40;
primaryTargetSensitivity = 0.90;

%% 2. Outer-training-only calibration
selectedAlertThreshold = nan(numberOfCases,numel(wearThresholds), ...
    numel(targetSensitivityGrid));
innerPositiveN = nan(size(selectedAlertThreshold));
innerSensitivity = nan(size(selectedAlertThreshold));
innerSpecificity = nan(size(selectedAlertThreshold));

calibratedAlert = false(numberOfSamples,numel(wearThresholds), ...
    numel(targetSensitivityGrid));
uncalibratedAlert = false(numberOfSamples,numel(wearThresholds));

fprintf("========== Training-only high-wear alert calibration ==========\n");
for caseIndex = 1:numberOfCases
    testCase = caseList(caseIndex);
    testMask = caseID == testCase;
    trainMask = ~testMask;

    hyperRow = hyperparameters.TestCase == testCase;
    assert(sum(hyperRow) == 1,"Missing outer-fold hyperparameters.");
    lambdaAll = hyperparameters.TimeAllLambda(hyperRow);
    lambdaExcluded = hyperparameters.TimeQualityLambda(hyperRow);
    gateThreshold = hyperparameters.TimeGateThreshold(hyperRow);

    % Inner OOF predictions are generated only within the outer-training
    % cases, with the outer-fold-selected ridge and gate parameters.
    innerPrediction = nan(numberOfSamples,1);
    trainingCases = unique(caseID(trainMask));
    for innerIndex = 1:numel(trainingCases)
        validationCase = trainingCases(innerIndex);
        validationMask = trainMask & caseID == validationCase;
        fittingMask = trainMask & caseID ~= validationCase;

        predictionAll = fitRidgeAndPredict( ...
            XAllTime(fittingMask,:),y(fittingMask), ...
            XAllTime(validationMask,:),lambdaAll);
        predictionExcluded = fitRidgeAndPredict( ...
            XExcludedTime(fittingMask,:),y(fittingMask), ...
            XExcludedTime(validationMask,:),lambdaExcluded);
        predictionGated = predictionAll;
        switchMask = saturationRatio(validationMask) >= gateThreshold;
        predictionGated(switchMask) = predictionExcluded(switchMask);
        innerPrediction(validationMask) = predictionGated;
    end
    assert(all(isfinite(innerPrediction(trainMask))), ...
        "Incomplete inner OOF predictions.");

    for wearIndex = 1:numel(wearThresholds)
        wearThreshold = wearThresholds(wearIndex);
        uncalibratedAlert(testMask,wearIndex) = ...
            primaryPrediction(testMask) >= wearThreshold;

        innerLabel = y(trainMask) >= wearThreshold;
        for targetIndex = 1:numel(targetSensitivityGrid)
            targetSensitivity = targetSensitivityGrid(targetIndex);
            [alertThreshold,achievedSensitivity,achievedSpecificity] = ...
                chooseSensitivityControlledThreshold( ...
                innerPrediction(trainMask),innerLabel, ...
                wearThreshold,targetSensitivity);

            selectedAlertThreshold(caseIndex,wearIndex,targetIndex) = ...
                alertThreshold;
            innerPositiveN(caseIndex,wearIndex,targetIndex) = ...
                sum(innerLabel);
            innerSensitivity(caseIndex,wearIndex,targetIndex) = ...
                achievedSensitivity;
            innerSpecificity(caseIndex,wearIndex,targetIndex) = ...
                achievedSpecificity;
            calibratedAlert(testMask,wearIndex,targetIndex) = ...
                primaryPrediction(testMask) >= alertThreshold;
        end
    end

    primaryWearIndex = find(wearThresholds == primaryWearThreshold,1);
    primaryTargetIndex = find( ...
        targetSensitivityGrid == primaryTargetSensitivity,1);
    fprintf("Case %d: alert score threshold %.4f for VB >= %.2f " + ...
        "(inner target %.0f%%)\n",testCase, ...
        selectedAlertThreshold(caseIndex,primaryWearIndex, ...
        primaryTargetIndex),primaryWearThreshold, ...
        100*primaryTargetSensitivity);
end

%% 3. Overall operating characteristics
operatingRows = {};
for wearIndex = 1:numel(wearThresholds)
    wearThreshold = wearThresholds(wearIndex);
    actualPositive = y >= wearThreshold;

    baseMetrics = classificationMetrics( ...
        actualPositive,uncalibratedAlert(:,wearIndex));
    operatingRows(end+1,:) = {wearThreshold,"Uncalibrated",NaN, ...
        wearThreshold,baseMetrics{:}}; %#ok<SAGROW>

    for targetIndex = 1:numel(targetSensitivityGrid)
        currentMetrics = classificationMetrics(actualPositive, ...
            calibratedAlert(:,wearIndex,targetIndex));
        operatingRows(end+1,:) = {wearThreshold,"Sensitivity calibrated", ...
            targetSensitivityGrid(targetIndex), ...
            mean(selectedAlertThreshold(:,wearIndex,targetIndex)), ...
            currentMetrics{:}}; %#ok<SAGROW>
    end
end

operatingResults = cell2table(operatingRows,'VariableNames', ...
    {'WearThreshold','AlertRule','TargetInnerSensitivity', ...
    'MeanSelectedScoreThreshold','TP','TN','FP','FN','Sensitivity', ...
    'Specificity','Precision','F1','BalancedAccuracy','MissRate', ...
    'PositiveRecordN','NegativeRecordN'});
operatingResults.AlertRule = string(operatingResults.AlertRule);

%% 4. Case-level paired inference for the primary alert endpoint
primaryWearIndex = find(wearThresholds == primaryWearThreshold,1);
primaryTargetIndex = find( ...
    targetSensitivityGrid == primaryTargetSensitivity,1);
basePrimaryAlert = uncalibratedAlert(:,primaryWearIndex);
calibratedPrimaryAlert = calibratedAlert(:,primaryWearIndex, ...
    primaryTargetIndex);
actualPrimaryPositive = y >= primaryWearThreshold;

casePositiveN = nan(numberOfCases,1);
caseNegativeN = nan(numberOfCases,1);
baseMissRate = nan(numberOfCases,1);
calibratedMissRate = nan(numberOfCases,1);
baseFalsePositiveRate = nan(numberOfCases,1);
calibratedFalsePositiveRate = nan(numberOfCases,1);

for caseIndex = 1:numberOfCases
    caseMask = caseID == caseList(caseIndex);
    positiveMask = caseMask & actualPrimaryPositive;
    negativeMask = caseMask & ~actualPrimaryPositive;
    casePositiveN(caseIndex) = sum(positiveMask);
    caseNegativeN(caseIndex) = sum(negativeMask);
    if any(positiveMask)
        baseMissRate(caseIndex) = mean(~basePrimaryAlert(positiveMask));
        calibratedMissRate(caseIndex) = ...
            mean(~calibratedPrimaryAlert(positiveMask));
    end
    if any(negativeMask)
        baseFalsePositiveRate(caseIndex) = ...
            mean(basePrimaryAlert(negativeMask));
        calibratedFalsePositiveRate(caseIndex) = ...
            mean(calibratedPrimaryAlert(negativeMask));
    end
end

caseResults = table(caseList,casePositiveN,caseNegativeN, ...
    baseMissRate,calibratedMissRate,baseFalsePositiveRate, ...
    calibratedFalsePositiveRate, ...
    selectedAlertThreshold(:,primaryWearIndex,primaryTargetIndex), ...
    'VariableNames',{'CaseID','PositiveRecordN','NegativeRecordN', ...
    'UncalibratedMissRate','CalibratedMissRate', ...
    'UncalibratedFalsePositiveRate','CalibratedFalsePositiveRate', ...
    'SelectedScoreThreshold'});

validMiss = isfinite(baseMissRate) & isfinite(calibratedMissRate);
validFalsePositive = isfinite(baseFalsePositiveRate) & ...
    isfinite(calibratedFalsePositiveRate);
missDifference = calibratedMissRate(validMiss)-baseMissRate(validMiss);
falsePositiveDifference = calibratedFalsePositiveRate(validFalsePositive)- ...
    baseFalsePositiveRate(validFalsePositive);

[missP,missStatistic] = safeSignrank(missDifference);
[falsePositiveP,falsePositiveStatistic] = ...
    safeSignrank(falsePositiveDifference);
rawP = [missP;falsePositiveP];
holmP = holmAdjust(rawP);
[missLower,missUpper,missImproveProbability] = ...
    bootstrapCaseDifference(missDifference,20267101);
[falsePositiveLower,falsePositiveUpper,falsePositiveImproveProbability] = ...
    bootstrapCaseDifference(falsePositiveDifference,20267102);

statisticalResults = table( ...
    ["Alert miss rate";"False-positive rate"], ...
    [sum(validMiss);sum(validFalsePositive)], ...
    [mean(missDifference);mean(falsePositiveDifference)], ...
    [missLower;falsePositiveLower],[missUpper;falsePositiveUpper], ...
    rawP,holmP,[missStatistic;falsePositiveStatistic], ...
    [missImproveProbability;falsePositiveImproveProbability], ...
    'VariableNames',{'Endpoint','IndependentCaseN', ...
    'MeanDifferenceCalibratedMinusUncalibrated', ...
    'BootstrapCILower','BootstrapCIUpper','RawWilcoxonP', ...
    'HolmAdjustedP','SignedRankStatistic', ...
    'BootstrapProbabilityDifferenceBelowZero'});

%% 5. Threshold-selection audit and prediction source data
selectionRows = {};
for caseIndex = 1:numberOfCases
    for wearIndex = 1:numel(wearThresholds)
        for targetIndex = 1:numel(targetSensitivityGrid)
            selectionRows(end+1,:) = {caseList(caseIndex), ...
                wearThresholds(wearIndex), ...
                targetSensitivityGrid(targetIndex), ...
                selectedAlertThreshold(caseIndex,wearIndex,targetIndex), ...
                innerPositiveN(caseIndex,wearIndex,targetIndex), ...
                innerSensitivity(caseIndex,wearIndex,targetIndex), ...
                innerSpecificity(caseIndex,wearIndex,targetIndex)}; %#ok<SAGROW>
        end
    end
end
selectionResults = cell2table(selectionRows,'VariableNames', ...
    {'OuterTestCase','WearThreshold','TargetInnerSensitivity', ...
    'SelectedScoreThreshold','InnerPositiveRecordN', ...
    'AchievedInnerSensitivity','AchievedInnerSpecificity'});

predictionSource = table(T.OriginalIndex,caseID,T.RunID,y, ...
    primaryPrediction,basePrimaryAlert,calibratedPrimaryAlert, ...
    'VariableNames',{'OriginalIndex','CaseID','RunID','ActualVB', ...
    'SQTRPrediction','UncalibratedVB04Alert', ...
    'CalibratedVB04Alert'});

%% 6. Save
writetable(operatingResults,fullfile(outputFolder, ...
    "alert_operating_characteristics.csv"));
writetable(caseResults,fullfile(outputFolder, ...
    "alert_case_results.csv"));
writetable(statisticalResults,fullfile(outputFolder, ...
    "alert_statistical_results.csv"));
writetable(selectionResults,fullfile(outputFolder, ...
    "alert_threshold_selections.csv"));
writetable(predictionSource,fullfile(outputFolder, ...
    "alert_prediction_source_data.csv"));

save(fullfile(outputFolder,"high_wear_alert_calibration.mat"), ...
    "operatingResults","caseResults","statisticalResults", ...
    "selectionResults","predictionSource","wearThresholds", ...
    "targetSensitivityGrid","primaryWearThreshold", ...
    "primaryTargetSensitivity","selectedAlertThreshold", ...
    "calibratedAlert","uncalibratedAlert");

fprintf("\n========== Alert operating characteristics ==========\n");
disp(operatingResults);
fprintf("\n========== Primary paired statistics ==========\n");
disp(statisticalResults);
fprintf("\nSaved to:\n%s\n",outputFolder);


function [threshold,sensitivity,specificity] = ...
    chooseSensitivityControlledThreshold( ...
    score,label,wearThreshold,targetSensitivity)

score = score(:);
label = logical(label(:));
assert(any(label),"No positive training records for alert calibration.");
candidateThreshold = unique([wearThreshold;score(score <= wearThreshold)]);
candidateThreshold = sort(candidateThreshold,"descend");
threshold = min(score)-max(eps(max(abs(score))),1e-12);

for candidateIndex = 1:numel(candidateThreshold)
    currentThreshold = candidateThreshold(candidateIndex);
    currentSensitivity = mean(score(label) >= currentThreshold);
    if currentSensitivity + 1e-12 >= targetSensitivity
        threshold = currentThreshold;
        break;
    end
end

prediction = score >= threshold;
sensitivity = mean(prediction(label));
if any(~label)
    specificity = mean(~prediction(~label));
else
    specificity = NaN;
end
end


function metrics = classificationMetrics(actualPositive,predictedPositive)
actualPositive = logical(actualPositive(:));
predictedPositive = logical(predictedPositive(:));
TP = sum(actualPositive & predictedPositive);
TN = sum(~actualPositive & ~predictedPositive);
FP = sum(~actualPositive & predictedPositive);
FN = sum(actualPositive & ~predictedPositive);
sensitivity = safeDivide(TP,TP+FN);
specificity = safeDivide(TN,TN+FP);
precision = safeDivide(TP,TP+FP);
F1 = safeDivide(2*precision*sensitivity,precision+sensitivity);
balancedAccuracy = mean([sensitivity specificity],"omitnan");
missRate = safeDivide(FN,TP+FN);
metrics = {TP,TN,FP,FN,sensitivity,specificity,precision,F1, ...
    balancedAccuracy,missRate,sum(actualPositive),sum(~actualPositive)};
end


function value = safeDivide(numerator,denominator)
if denominator == 0
    value = NaN;
else
    value = numerator/denominator;
end
end


function [p,statistic] = safeSignrank(difference)
difference = difference(isfinite(difference));
if isempty(difference) || all(abs(difference) < 1e-15)
    p = 1;
    statistic = 0;
    return;
end
[p,~,stats] = signrank(difference,0,"method","exact");
if isfield(stats,"signedrank")
    statistic = stats.signedrank;
else
    statistic = NaN;
end
end


function adjusted = holmAdjust(p)
[sortedP,order] = sort(p(:));
numberOfTests = numel(sortedP);
sortedAdjusted = nan(numberOfTests,1);
runningMaximum = 0;
for index = 1:numberOfTests
    currentAdjusted = (numberOfTests-index+1)*sortedP(index);
    runningMaximum = max(runningMaximum,currentAdjusted);
    sortedAdjusted(index) = min(runningMaximum,1);
end
adjusted = nan(numberOfTests,1);
adjusted(order) = sortedAdjusted;
end


function [lower,upper,probability] = ...
    bootstrapCaseDifference(difference,seed)
difference = difference(isfinite(difference));
numberOfCases = numel(difference);
stream = RandStream("mt19937ar","Seed",seed);
bootstrapMean = nan(10000,1);
for bootstrapIndex = 1:10000
    sampledIndex = randi(stream,numberOfCases,numberOfCases,1);
    bootstrapMean(bootstrapIndex) = mean(difference(sampledIndex));
end
lower = prctile(bootstrapMean,2.5);
upper = prctile(bootstrapMean,97.5);
probability = mean(bootstrapMean < 0);
end
