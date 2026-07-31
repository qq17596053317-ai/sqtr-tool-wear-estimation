paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. Load data and helpers
scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);

assert(exist("groupedRidgePrediction","file") == 2, ...
    "groupedRidgePrediction.m was not found.");
assert(exist("groupedGatedRidgePrediction","file") == 2, ...
    "groupedGatedRidgePrediction.m was not found.");

resultFolder = fullfile(paperRoot, 'mill', 'results');
timeData = load(fullfile(resultFolder,"quality_aware_with_time.mat"));
T = timeData.timeAwareTable;

%% 2. Variables and model inputs
y = T.VB;
caseID = T.CaseID;
runID = T.RunID;
originalIndex = T.OriginalIndex;
elapsedTime = T.ElapsedTime;
elapsedTimeSquared = elapsedTime.^2;
saturationRatio = T.smcDC_SaturationRatio;

caseList = unique(caseID);
numberOfCases = numel(caseList);
numberOfSamples = height(T);

allVariableNames = string(T.Properties.VariableNames);
sensorVariableNames = allVariableNames(8:85);
smcDCMask = startsWith(sensorVariableNames,"smcDC_");
otherSensorNames = sensorVariableNames(~smcDCMask);

XProcess = T{:,{'DOC','Feed','Material'}};
XAllSensors = T{:,cellstr(sensorVariableNames)};
XOtherSensors = T{:,cellstr(otherSensorNames)};

XTimeOnly = [XProcess,elapsedTime,elapsedTimeSquared];
XAllSensorsNoTime = [XProcess,XAllSensors];
XQualityAwareNoTime = [XProcess,XOtherSensors];
XAllSensorsTime = [XProcess,XAllSensors,elapsedTime,elapsedTimeSquared];
XQualityAwareTime = [XProcess,XOtherSensors, ...
    elapsedTime,elapsedTimeSquared];

fprintf("========== Input dimensions ==========\n");
fprintf("TimeOnly: %d\n",size(XTimeOnly,2));
fprintf("AllSensorsNoTime: %d\n",size(XAllSensorsNoTime,2));
fprintf("QualityAwareNoTime: %d\n",size(XQualityAwareNoTime,2));
fprintf("AllSensorsTime: %d\n",size(XAllSensorsTime,2));
fprintf("QualityAwareTime: %d\n",size(XQualityAwareTime,2));

%% 3. Strict nested leave-one-case-out validation
lambdaGrid = logspace(-4,4,17);
% Prespecified quality-gate candidates; never selected from outer test data.
gateThresholdGrid = [0.01 0.10 0.50];

predRidgeAll = nan(numberOfSamples,1);
predRidgeQuality = nan(numberOfSamples,1);
predRidgeGated = nan(numberOfSamples,1);
predTimeOnly = nan(numberOfSamples,1);
predAllSensorsTime = nan(numberOfSamples,1);
predQualityAwareTime = nan(numberOfSamples,1);
predTimeGated = nan(numberOfSamples,1);

bestLambdaTimeOnly = nan(numberOfCases,1);
bestLambdaRidgeAll = nan(numberOfCases,1);
bestLambdaRidgeQuality = nan(numberOfCases,1);
bestLambdaTimeAll = nan(numberOfCases,1);
bestLambdaTimeQuality = nan(numberOfCases,1);
bestGateThresholdRidge = nan(numberOfCases,1);
bestGateThresholdTime = nan(numberOfCases,1);
gateValidationMAERidge = nan(numberOfCases,numel(gateThresholdGrid));
gateValidationMAETime = nan(numberOfCases,numel(gateThresholdGrid));

for caseIndex = 1:numberOfCases
    testCase = caseList(caseIndex);
    testMask = caseID == testCase;
    trainMask = ~testMask;

    fprintf("\nOuter test Case %d: train %d, test %d\n", ...
        testCase,sum(trainMask),sum(testMask));

    [predTimeOnly(testMask),bestLambdaTimeOnly(caseIndex)] = ...
        groupedRidgePrediction(XTimeOnly,y,caseID, ...
        trainMask,testMask,lambdaGrid);

    [predRidgeAll(testMask),predRidgeQuality(testMask), ...
        predRidgeGated(testMask),bestLambdaRidgeAll(caseIndex), ...
        bestLambdaRidgeQuality(caseIndex), ...
        bestGateThresholdRidge(caseIndex), ...
        gateValidationMAERidge(caseIndex,:)] = ...
        groupedGatedRidgePrediction( ...
        XAllSensorsNoTime,XQualityAwareNoTime,saturationRatio, ...
        y,caseID,trainMask,testMask,lambdaGrid,gateThresholdGrid);

    [predAllSensorsTime(testMask),predQualityAwareTime(testMask), ...
        predTimeGated(testMask),bestLambdaTimeAll(caseIndex), ...
        bestLambdaTimeQuality(caseIndex), ...
        bestGateThresholdTime(caseIndex), ...
        gateValidationMAETime(caseIndex,:)] = ...
        groupedGatedRidgePrediction( ...
        XAllSensorsTime,XQualityAwareTime,saturationRatio, ...
        y,caseID,trainMask,testMask,lambdaGrid,gateThresholdGrid);

    fprintf("  No-time gate: %.0f%%; time gate: %.0f%%\n", ...
        100*bestGateThresholdRidge(caseIndex), ...
        100*bestGateThresholdTime(caseIndex));
end

assert(all(isfinite([predRidgeGated;predTimeOnly;predAllSensorsTime; ...
    predQualityAwareTime;predTimeGated])),"Non-finite predictions found.");

%% 4. Overall evaluation
modelNames = ["RidgeGated","TimeOnly","AllSensorsTime", ...
    "QualityAwareTime","TimeGated"];
predictionMatrix = [predRidgeGated,predTimeOnly,predAllSensorsTime, ...
    predQualityAwareTime,predTimeGated];
numberOfModels = numel(modelNames);

MAE = nan(numberOfModels,1);
RMSE = nan(numberOfModels,1);
R2 = nan(numberOfModels,1);
P95AbsoluteError = nan(numberOfModels,1);
MaximumAbsoluteError = nan(numberOfModels,1);

for modelIndex = 1:numberOfModels
    residual = predictionMatrix(:,modelIndex)-y;
    absoluteError = abs(residual);
    MAE(modelIndex) = mean(absoluteError);
    RMSE(modelIndex) = sqrt(mean(residual.^2));
    R2(modelIndex) = 1-sum(residual.^2)/sum((y-mean(y)).^2);
    P95AbsoluteError(modelIndex) = prctile(absoluteError,95);
    MaximumAbsoluteError(modelIndex) = max(absoluteError);
end

overallResults = table(modelNames',MAE,RMSE,R2,P95AbsoluteError, ...
    MaximumAbsoluteError,'VariableNames',{'Model','MAE','RMSE','R2', ...
    'P95AbsoluteError','MaximumAbsoluteError'});
fprintf("\n========== Strict nested LOCO results ==========\n");
disp(overallResults);

%% 5. Case-level results
caseSampleCount = nan(numberOfCases,1);
caseMAE = nan(numberOfCases,numberOfModels);
for caseIndex = 1:numberOfCases
    currentMask = caseID == caseList(caseIndex);
    caseSampleCount(caseIndex) = sum(currentMask);
    for modelIndex = 1:numberOfModels
        caseMAE(caseIndex,modelIndex) = mean(abs( ...
            predictionMatrix(currentMask,modelIndex)-y(currentMask)));
    end
end

caseResults = table(caseList,caseSampleCount,caseMAE(:,1),caseMAE(:,2), ...
    caseMAE(:,3),caseMAE(:,4),caseMAE(:,5),'VariableNames', ...
    {'CaseID','SampleCount','RidgeGatedMAE','TimeOnlyMAE', ...
    'AllSensorsTimeMAE','QualityAwareTimeMAE','TimeGatedMAE'});
fprintf("\n========== Case-level results ==========\n");
disp(caseResults);

%% 6. Worst errors and selected hyperparameters
WorstOriginalIndex = nan(numberOfModels,1);
WorstCaseID = nan(numberOfModels,1);
WorstRunID = nan(numberOfModels,1);
WorstActualVB = nan(numberOfModels,1);
WorstPredictedVB = nan(numberOfModels,1);
WorstAbsoluteError = nan(numberOfModels,1);

for modelIndex = 1:numberOfModels
    [WorstAbsoluteError(modelIndex),rowIndex] = max(abs( ...
        predictionMatrix(:,modelIndex)-y));
    WorstOriginalIndex(modelIndex) = originalIndex(rowIndex);
    WorstCaseID(modelIndex) = caseID(rowIndex);
    WorstRunID(modelIndex) = runID(rowIndex);
    WorstActualVB(modelIndex) = y(rowIndex);
    WorstPredictedVB(modelIndex) = predictionMatrix(rowIndex,modelIndex);
end

worstResults = table(modelNames',WorstOriginalIndex,WorstCaseID, ...
    WorstRunID,WorstActualVB,WorstPredictedVB,WorstAbsoluteError, ...
    'VariableNames',{'Model','OriginalIndex','CaseID','RunID', ...
    'ActualVB','PredictedVB','AbsoluteError'});

thresholdSelectionTable = table(caseList,bestLambdaTimeOnly, ...
    bestLambdaRidgeAll,bestLambdaRidgeQuality,bestGateThresholdRidge, ...
    bestLambdaTimeAll,bestLambdaTimeQuality,bestGateThresholdTime, ...
    'VariableNames',{'TestCase','TimeOnlyLambda','RidgeAllLambda', ...
    'RidgeQualityLambda','RidgeGateThreshold','TimeAllLambda', ...
    'TimeQualityLambda','TimeGateThreshold'});

for thresholdIndex = 1:numel(gateThresholdGrid)
    suffix = sprintf('Gate%03d',round(100*gateThresholdGrid(thresholdIndex)));
    thresholdSelectionTable.("RidgeValidationMAE_"+suffix) = ...
        gateValidationMAERidge(:,thresholdIndex);
    thresholdSelectionTable.("TimeValidationMAE_"+suffix) = ...
        gateValidationMAETime(:,thresholdIndex);
end

fprintf("\n========== Training-only gate selections ==========\n");
disp(thresholdSelectionTable(:,{'TestCase','RidgeGateThreshold', ...
    'TimeGateThreshold'}));

%% 7. Figures
fig1 = figure("Color","w","Position",[100 100 1500 600]);
bar(categorical(modelNames),[MAE RMSE]);
grid on;
ylabel("Error");
legend("MAE","RMSE","Location","northwest");
title("Strict Nested LOCO Comparison of Time-Augmented Models");

fig2 = figure("Color","w","Position",[100 100 1500 650]);
bar(caseList,caseMAE,"grouped");
grid on;
xlabel("Held-out case");
ylabel("MAE");
title("Case-Level Performance with Training-Selected Quality Gates");
legend(modelNames,"Location","northwest");

for fig = [fig1 fig2]
    axesHandles = findall(fig,"Type","axes");
    for axisIndex = 1:numel(axesHandles)
        if ~isempty(axesHandles(axisIndex).Toolbar)
            axesHandles(axisIndex).Toolbar.Visible = "off";
        end
        disableDefaultInteractivity(axesHandles(axisIndex));
    end
end
drawnow;

%% 8. Save
predictionTable = table(originalIndex,caseID,runID,y,elapsedTime, ...
    saturationRatio,predRidgeGated,predTimeOnly,predAllSensorsTime, ...
    predQualityAwareTime,predTimeGated,'VariableNames', ...
    {'OriginalIndex','CaseID','RunID','ActualVB','ElapsedTime', ...
    'SaturationRatio','PredRidgeGated','PredTimeOnly', ...
    'PredAllSensorsTime','PredQualityAwareTime','PredTimeGated'});

writetable(overallResults,fullfile(resultFolder,"time_model_overall.csv"));
writetable(caseResults,fullfile(resultFolder,"time_model_cases.csv"));
writetable(worstResults,fullfile(resultFolder,"time_model_worst.csv"));
writetable(predictionTable,fullfile(resultFolder,"time_model_predictions.csv"));
writetable(thresholdSelectionTable,fullfile(resultFolder, ...
    "time_model_nested_hyperparameters.csv"));

bestLambda = [bestLambdaTimeOnly,bestLambdaTimeAll,bestLambdaTimeQuality];
save(fullfile(resultFolder,"time_model_results.mat"), ...
    "overallResults","caseResults","worstResults","predictionTable", ...
    "thresholdSelectionTable","bestLambda","bestLambdaTimeOnly", ...
    "bestLambdaRidgeAll","bestLambdaRidgeQuality", ...
    "bestLambdaTimeAll","bestLambdaTimeQuality", ...
    "bestGateThresholdRidge","bestGateThresholdTime", ...
    "gateThresholdGrid","gateValidationMAERidge", ...
    "gateValidationMAETime");

exportgraphics(fig1,fullfile(resultFolder,"time_model_overall.png"), ...
    "Resolution",300);
exportgraphics(fig2,fullfile(resultFolder,"time_model_case_MAE.png"), ...
    "Resolution",300);

fprintf("\nSaved strict nested main-model results to:\n%s\n",resultFolder);
