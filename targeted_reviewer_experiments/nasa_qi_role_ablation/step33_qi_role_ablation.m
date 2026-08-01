clear;
clc;
close all;

%% Paths and inputs
scriptFolder = fileparts(mfilename("fullpath"));
projectFolder = fileparts(fileparts(scriptFolder));
resultFolder = fullfile(projectFolder,"mill","results");
outputFolder = scriptFolder;
if ~exist(outputFolder,"dir")
    mkdir(outputFolder);
end
addpath(projectFolder);

loaded = load(fullfile(resultFolder,"quality_aware_with_time.mat"));
T = loaded.timeAwareTable;

y = T.VB;
caseID = T.CaseID;
runID = T.RunID;
originalIndex = T.OriginalIndex;
elapsedTime = T.ElapsedTime;
elapsedTimeSquared = elapsedTime.^2;
qualityRatioFixed = T.smcDC_SaturationRatio;

allVariableNames = string(T.Properties.VariableNames);
sensorVariableNames = allVariableNames(8:85);
smcDCMask = startsWith(sensorVariableNames,"smcDC_");
otherSensorNames = sensorVariableNames(~smcDCMask);

XProcess = T{:,{'DOC','Feed','Material'}};
XAllSensors = T{:,cellstr(sensorVariableNames)};
XOtherSensors = T{:,cellstr(otherSensorNames)};

XProcessTime = [XProcess,elapsedTime,elapsedTimeSquared];
XProcessTimeQ = [XProcess,elapsedTime,elapsedTimeSquared,qualityRatioFixed];
XAllTime = [XProcess,XAllSensors,elapsedTime,elapsedTimeSquared];
XAllTimeQ = [XProcess,XAllSensors,elapsedTime,elapsedTimeSquared,qualityRatioFixed];
XDropTime = [XProcess,XOtherSensors,elapsedTime,elapsedTimeSquared];
XDropTimeQ = [XProcess,XOtherSensors,elapsedTime,elapsedTimeSquared,qualityRatioFixed];

modelNames = [ ...
    "ProcessTime", ...
    "ProcessTimeQ", ...
    "AllTime", ...
    "AllTimeQ", ...
    "DropTime", ...
    "DropTimeQ", ...
    "RouteOnly", ...
    "RoutePlusQ"];
numberOfModels = numel(modelNames);

caseList = unique(caseID);
numberOfCases = numel(caseList);
numberOfSamples = height(T);
lambdaGrid = logspace(-4,4,17);
gateThresholdGrid = [0.01 0.10 0.50];

predictions = nan(numberOfSamples,numberOfModels);
selectedLambda = nan(numberOfCases,6);
selectedGate = nan(numberOfCases,2);
gateValidationMAERouteOnly = nan(numberOfCases,numel(gateThresholdGrid));
gateValidationMAERoutePlusQ = nan(numberOfCases,numel(gateThresholdGrid));

%% Strict nested LOCO for every q-role variant
for caseIndex = 1:numberOfCases
    testCase = caseList(caseIndex);
    testMask = caseID == testCase;
    trainMask = ~testMask;

    [predictions(testMask,1),selectedLambda(caseIndex,1)] = ...
        groupedRidgePrediction(XProcessTime,y,caseID, ...
        trainMask,testMask,lambdaGrid);
    [predictions(testMask,2),selectedLambda(caseIndex,2)] = ...
        groupedRidgePrediction(XProcessTimeQ,y,caseID, ...
        trainMask,testMask,lambdaGrid);
    [predictions(testMask,3),selectedLambda(caseIndex,3)] = ...
        groupedRidgePrediction(XAllTime,y,caseID, ...
        trainMask,testMask,lambdaGrid);
    [predictions(testMask,4),selectedLambda(caseIndex,4)] = ...
        groupedRidgePrediction(XAllTimeQ,y,caseID, ...
        trainMask,testMask,lambdaGrid);
    [predictions(testMask,5),selectedLambda(caseIndex,5)] = ...
        groupedRidgePrediction(XDropTime,y,caseID, ...
        trainMask,testMask,lambdaGrid);
    [predictions(testMask,6),selectedLambda(caseIndex,6)] = ...
        groupedRidgePrediction(XDropTimeQ,y,caseID, ...
        trainMask,testMask,lambdaGrid);

    [~,~,predictions(testMask,7),~,~,selectedGate(caseIndex,1), ...
        gateValidationMAERouteOnly(caseIndex,:)] = ...
        groupedGatedRidgePrediction( ...
        XAllTime,XDropTime,qualityRatioFixed,y,caseID, ...
        trainMask,testMask,lambdaGrid,gateThresholdGrid);

    [~,~,predictions(testMask,8),~,~,selectedGate(caseIndex,2), ...
        gateValidationMAERoutePlusQ(caseIndex,:)] = ...
        groupedGatedRidgePrediction( ...
        XAllTime,XDropTimeQ,qualityRatioFixed,y,caseID, ...
        trainMask,testMask,lambdaGrid,gateThresholdGrid);
end

assert(all(isfinite(predictions),"all"),"Non-finite predictions found.");

%% Overall and case-level metrics
MAE = nan(numberOfModels,1);
RMSE = nan(numberOfModels,1);
R2 = nan(numberOfModels,1);
P95AbsoluteError = nan(numberOfModels,1);
MaximumAbsoluteError = nan(numberOfModels,1);
for modelIndex = 1:numberOfModels
    residual = predictions(:,modelIndex)-y;
    absoluteError = abs(residual);
    MAE(modelIndex) = mean(absoluteError);
    RMSE(modelIndex) = sqrt(mean(residual.^2));
    R2(modelIndex) = 1-sum(residual.^2)/sum((y-mean(y)).^2);
    P95AbsoluteError(modelIndex) = prctile(absoluteError,95);
    MaximumAbsoluteError(modelIndex) = max(absoluteError);
end

caseMAE = nan(numberOfCases,numberOfModels);
caseSampleCount = nan(numberOfCases,1);
for caseIndex = 1:numberOfCases
    currentMask = caseID == caseList(caseIndex);
    caseSampleCount(caseIndex) = sum(currentMask);
    caseMAE(caseIndex,:) = mean(abs( ...
        predictions(currentMask,:)-y(currentMask)),1);
end
macroMAE = mean(caseMAE,1)';

overallResults = table(modelNames',MAE,RMSE,R2,P95AbsoluteError, ...
    MaximumAbsoluteError,macroMAE, ...
    'VariableNames',{'Model','MAE','RMSE','R2','P95AbsoluteError', ...
    'MaximumAbsoluteError','MacroMAE'});

caseResults = table(caseList,caseSampleCount);
for modelIndex = 1:numberOfModels
    caseResults.(modelNames(modelIndex)+"MAE") = caseMAE(:,modelIndex);
end

%% Prespecified q-role comparisons at the independent-case level
comparisonNames = [ ...
    "ProcessTimeQ_vs_ProcessTime"; ...
    "AllTimeQ_vs_AllTime"; ...
    "DropTimeQ_vs_DropTime"; ...
    "RoutePlusQ_vs_RouteOnly"];
modelA = [1;3;5;7];
modelB = [2;4;6;8];
numberOfComparisons = numel(comparisonNames);

pooledDifferenceBMinusA = MAE(modelB)-MAE(modelA);
macroDifferenceBMinusA = macroMAE(modelB)-macroMAE(modelA);
winCountB = sum(caseMAE(:,modelB) < caseMAE(:,modelA),1)';
rawP = nan(numberOfComparisons,1);
ciLower = nan(numberOfComparisons,1);
ciUpper = nan(numberOfComparisons,1);

rng(20260729,"twister");
numberOfBootstrapSamples = 10000;
bootstrapIndices = randi(numberOfCases, ...
    numberOfCases,numberOfBootstrapSamples);

for comparisonIndex = 1:numberOfComparisons
    differences = caseMAE(:,modelB(comparisonIndex)) - ...
        caseMAE(:,modelA(comparisonIndex));
    rawP(comparisonIndex) = signrank(differences,0, ...
        "tail","both","method","exact");
    bootstrapMeans = mean(differences(bootstrapIndices),1);
    ciLower(comparisonIndex) = prctile(bootstrapMeans,2.5);
    ciUpper(comparisonIndex) = prctile(bootstrapMeans,97.5);
end
holmP = holmAdjust(rawP);

comparisonResults = table(comparisonNames, ...
    modelNames(modelA)',modelNames(modelB)', ...
    pooledDifferenceBMinusA,macroDifferenceBMinusA, ...
    ciLower,ciUpper,winCountB,rawP,holmP, ...
    'VariableNames',{'Comparison','ModelA','ModelB', ...
    'PooledMAEDifferenceBMinusA','MacroMAEDifferenceBMinusA', ...
    'MacroDifferenceCILower','MacroDifferenceCIUpper', ...
    'CasesImprovedByB','RawP','HolmP'});

%% Additional route-effect comparisons
routeComparisons = [ ...
    "RouteOnly_vs_AllTime"; ...
    "RoutePlusQ_vs_AllTime"];
routeModelA = [3;3];
routeModelB = [7;8];
routeRawP = nan(2,1);
routePooledDifference = MAE(routeModelB)-MAE(routeModelA);
routeMacroDifference = macroMAE(routeModelB)-macroMAE(routeModelA);
routeCILower = nan(2,1);
routeCIUpper = nan(2,1);
routeWinCount = sum(caseMAE(:,routeModelB) < caseMAE(:,routeModelA),1)';
for comparisonIndex = 1:2
    differences = caseMAE(:,routeModelB(comparisonIndex)) - ...
        caseMAE(:,routeModelA(comparisonIndex));
    routeRawP(comparisonIndex) = signrank(differences,0, ...
        "tail","both","method","exact");
    bootstrapMeans = mean(differences(bootstrapIndices),1);
    routeCILower(comparisonIndex) = prctile(bootstrapMeans,2.5);
    routeCIUpper(comparisonIndex) = prctile(bootstrapMeans,97.5);
end
routeHolmP = holmAdjust(routeRawP);
routeComparisonResults = table(routeComparisons, ...
    modelNames(routeModelA)',modelNames(routeModelB)', ...
    routePooledDifference,routeMacroDifference, ...
    routeCILower,routeCIUpper,routeWinCount,routeRawP,routeHolmP, ...
    'VariableNames',{'Comparison','ModelA','ModelB', ...
    'PooledMAEDifferenceBMinusA','MacroMAEDifferenceBMinusA', ...
    'MacroDifferenceCILower','MacroDifferenceCIUpper', ...
    'CasesImprovedByB','RawP','HolmP'});

%% Save
predictionTable = table(originalIndex,caseID,runID,y,elapsedTime, ...
    qualityRatioFixed,'VariableNames',{'OriginalIndex','CaseID', ...
    'RunID','ActualVB','ElapsedTime','QualityRatio'});
for modelIndex = 1:numberOfModels
    predictionTable.("Pred"+modelNames(modelIndex)) = predictions(:,modelIndex);
end

hyperparameterResults = table(caseList);
for modelIndex = 1:6
    hyperparameterResults.("Lambda"+modelNames(modelIndex)) = ...
        selectedLambda(:,modelIndex);
end
hyperparameterResults.RouteOnlyGate = selectedGate(:,1);
hyperparameterResults.RoutePlusQGate = selectedGate(:,2);

writetable(overallResults,fullfile(outputFolder,"qi_role_overall.csv"));
writetable(caseResults,fullfile(outputFolder,"qi_role_cases.csv"));
writetable(comparisonResults,fullfile(outputFolder,"qi_role_comparisons.csv"));
writetable(routeComparisonResults, ...
    fullfile(outputFolder,"qi_role_route_comparisons.csv"));
writetable(predictionTable,fullfile(outputFolder,"qi_role_predictions.csv"));
writetable(hyperparameterResults, ...
    fullfile(outputFolder,"qi_role_hyperparameters.csv"));

save(fullfile(outputFolder,"qi_role_ablation.mat"), ...
    "overallResults","caseResults","comparisonResults", ...
    "routeComparisonResults","predictionTable","hyperparameterResults", ...
    "predictions","caseMAE","selectedLambda","selectedGate", ...
    "gateThresholdGrid","gateValidationMAERouteOnly", ...
    "gateValidationMAERoutePlusQ");

disp("========== q-role ablation: overall ==========");
disp(overallResults);
disp("========== q-role comparisons ==========");
disp(comparisonResults);
disp("========== route comparisons ==========");
disp(routeComparisonResults);


function adjustedP = holmAdjust(rawP)
    numberOfTests = numel(rawP);
    [sortedP,order] = sort(rawP);
    sortedAdjusted = nan(size(sortedP));
    runningMaximum = 0;
    for rankIndex = 1:numberOfTests
        currentAdjusted = (numberOfTests-rankIndex+1)*sortedP(rankIndex);
        runningMaximum = max(runningMaximum,currentAdjusted);
        sortedAdjusted(rankIndex) = min(1,runningMaximum);
    end
    adjustedP = nan(size(rawP));
    adjustedP(order) = sortedAdjusted;
end
