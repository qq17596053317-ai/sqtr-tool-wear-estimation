paperRoot = fileparts(mfilename('fullpath'));
clear;
clc;
close all;

%% 1. Data and frozen nested-LOCO policy
scriptFolder = fileparts(mfilename("fullpath"));
addpath(scriptFolder);

resultFolder = fullfile(paperRoot, 'mill', 'results');
outputFolder = fullfile(paperRoot, 'additional_validation_results');
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

timeData = load(fullfile(resultFolder,"quality_aware_with_time.mat"));
modelData = load(fullfile(resultFolder,"time_model_results.mat"));
T = timeData.timeAwareTable;

requiredFields = ["bestLambdaTimeAll","bestLambdaTimeQuality", ...
    "bestGateThresholdTime","predictionTable"];
for fieldIndex = 1:numel(requiredFields)
    assert(isfield(modelData,requiredFields(fieldIndex)), ...
        "Rerun corrected step15 before step23; missing %s.", ...
        requiredFields(fieldIndex));
end

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

observedPrediction = modelData.predictionTable.PredTimeGated;
observedMAE = mean(abs(observedPrediction-y));
observedCaseMAE = caseMAEVector(y,observedPrediction,caseID,caseList);
observedMacroMAE = mean(observedCaseMAE);

%% 2. Case-wise time-permutation diagnostic
% The model policy (lambdas and gate threshold) is frozen from the corrected
% nested-LOCO analysis. Time is permuted within each case for both training
% and held-out records, preserving each case's marginal time distribution
% while breaking its alignment with wear progression.
numberOfPermutations = 100;
permutationMAE = nan(numberOfPermutations,1);
permutationMacroMAE = nan(numberOfPermutations,1);
permutationPredictions = nan(numberOfSamples,numberOfPermutations);

fprintf("========== Case-wise elapsed-time permutation ==========" + newline);
for repeatIndex = 1:numberOfPermutations
    permutedTime = elapsedTime;
    for caseIndex = 1:numberOfCases
        currentIndex = find(caseID == caseList(caseIndex));
        stream = RandStream("mt19937ar","Seed", ...
            20260718 + 1000*repeatIndex + caseIndex);
        order = randperm(stream,numel(currentIndex));
        permutedTime(currentIndex) = elapsedTime(currentIndex(order));
    end

    currentPrediction = frozenPolicyPrediction( ...
        XProcess,XAllSensors,XOtherSensors,permutedTime, ...
        saturationRatio,y,caseID,caseList,modelData);
    permutationPredictions(:,repeatIndex) = currentPrediction;
    permutationMAE(repeatIndex) = mean(abs(currentPrediction-y));
    permutationMacroMAE(repeatIndex) = mean( ...
        caseMAEVector(y,currentPrediction,caseID,caseList));

    if mod(repeatIndex,10) == 0
        fprintf("  completed %d/%d permutations\n", ...
            repeatIndex,numberOfPermutations);
    end
end

empiricalP = (1 + sum(permutationMAE <= observedMAE)) / ...
    (numberOfPermutations + 1);
observedPercentile = 100*mean(permutationMAE >= observedMAE);

permutationSummary = table(observedMAE,observedMacroMAE, ...
    mean(permutationMAE),std(permutationMAE), ...
    prctile(permutationMAE,2.5),prctile(permutationMAE,97.5), ...
    empiricalP,observedPercentile,numberOfPermutations, ...
    'VariableNames',{'ObservedMAE','ObservedMacroMAE', ...
    'PermutationMeanMAE','PermutationSDMAE','PermutationCI025', ...
    'PermutationCI975','EmpiricalP','ObservedBetterThanPercent', ...
    'PermutationCount'});

permutationRepeats = table((1:numberOfPermutations)', ...
    permutationMAE,permutationMacroMAE, ...
    'VariableNames',{'Repeat','MAE','MacroMAE'});

%% 3. Test-time timestamp uncertainty and missingness
scenarioNames = ["Clean","Noise5Percent","Noise10Percent", ...
    "Noise20Percent","Missing25Percent"];
noiseSD = [0,0.05,0.10,0.20,0];
missingRate = [0,0,0,0,0.25];
numberOfScenarios = numel(scenarioNames);
numberOfRobustnessRepeats = 50;

robustnessMAE = nan(numberOfRobustnessRepeats,numberOfScenarios);
robustnessMacroMAE = nan(numberOfRobustnessRepeats,numberOfScenarios);
caseScenarioMAE = nan(numberOfCases,numberOfScenarios, ...
    numberOfRobustnessRepeats);

fprintf("\n========== Test-time elapsed-time uncertainty ==========" + newline);
for scenarioIndex = 1:numberOfScenarios
    fprintf("  scenario: %s\n",scenarioNames(scenarioIndex));

    for repeatIndex = 1:numberOfRobustnessRepeats
        currentPrediction = nan(numberOfSamples,1);

        for caseIndex = 1:numberOfCases
            testCase = caseList(caseIndex);
            testMask = caseID == testCase;
            trainMask = ~testMask;
            testTime = elapsedTime(testMask);

            stream = RandStream("mt19937ar","Seed", ...
                20261718 + 100000*scenarioIndex + ...
                1000*repeatIndex + caseIndex);

            if noiseSD(scenarioIndex) > 0
                testTime = testTime .* (1 + ...
                    noiseSD(scenarioIndex)*randn(stream,size(testTime)));
                testTime = max(testTime,0);
            end

            if missingRate(scenarioIndex) > 0
                missingMask = rand(stream,size(testTime)) < ...
                    missingRate(scenarioIndex);
                trainingMedian = median(elapsedTime(trainMask));
                testTime(missingMask) = trainingMedian;
            end

            XAllTrain = [XProcess(trainMask,:),XAllSensors(trainMask,:), ...
                elapsedTime(trainMask),elapsedTime(trainMask).^2];
            XQualityTrain = [XProcess(trainMask,:), ...
                XOtherSensors(trainMask,:), ...
                elapsedTime(trainMask),elapsedTime(trainMask).^2];
            XAllTest = [XProcess(testMask,:),XAllSensors(testMask,:), ...
                testTime,testTime.^2];
            XQualityTest = [XProcess(testMask,:), ...
                XOtherSensors(testMask,:), ...
                testTime,testTime.^2];

            predAll = fitRidgeAndPredict(XAllTrain,y(trainMask),XAllTest, ...
                modelData.bestLambdaTimeAll(caseIndex));
            predQuality = fitRidgeAndPredict(XQualityTrain,y(trainMask), ...
                XQualityTest,modelData.bestLambdaTimeQuality(caseIndex));
            predGated = predAll;
            switchMask = saturationRatio(testMask) >= ...
                modelData.bestGateThresholdTime(caseIndex);
            predGated(switchMask) = predQuality(switchMask);
            currentPrediction(testMask) = predGated;
        end

        currentCaseMAE = caseMAEVector(y,currentPrediction,caseID,caseList);
        robustnessMAE(repeatIndex,scenarioIndex) = ...
            mean(abs(currentPrediction-y));
        robustnessMacroMAE(repeatIndex,scenarioIndex) = mean(currentCaseMAE);
        caseScenarioMAE(:,scenarioIndex,repeatIndex) = currentCaseMAE;
    end
end

Scenario = scenarioNames';
MeanMAE = mean(robustnessMAE,1)';
SDMAE = std(robustnessMAE,0,1)';
MeanMacroMAE = mean(robustnessMacroMAE,1)';
SDMacroMAE = std(robustnessMacroMAE,0,1)';
MAEChangePercent = 100*(MeanMAE-MeanMAE(1))/MeanMAE(1);

scenarioSummary = table(Scenario,MeanMAE,SDMAE,MeanMacroMAE, ...
    SDMacroMAE,MAEChangePercent);

repeatColumn = repelem((1:numberOfRobustnessRepeats)',numberOfScenarios);
scenarioColumn = repmat(scenarioNames',numberOfRobustnessRepeats,1);
robustnessRepeats = table(repeatColumn,scenarioColumn, ...
    reshape(robustnessMAE',[],1),reshape(robustnessMacroMAE',[],1), ...
    'VariableNames',{'Repeat','Scenario','MAE','MacroMAE'});

% Case is the independent unit. Repeated perturbations are averaged before
% paired inference. Four prespecified scenario-vs-clean tests use Holm.
meanCaseScenarioMAE = mean(caseScenarioMAE,3);
rawP = nan(numberOfScenarios-1,1);
signedRankStatistic = nan(numberOfScenarios-1,1);
meanPairedDifference = nan(numberOfScenarios-1,1);
bootstrapLower = nan(numberOfScenarios-1,1);
bootstrapUpper = nan(numberOfScenarios-1,1);

for comparisonIndex = 1:(numberOfScenarios-1)
    scenarioIndex = comparisonIndex+1;
    difference = meanCaseScenarioMAE(:,scenarioIndex) - ...
        meanCaseScenarioMAE(:,1);
    meanPairedDifference(comparisonIndex) = mean(difference);
    [rawP(comparisonIndex),~,stats] = signrank( ...
        meanCaseScenarioMAE(:,scenarioIndex),meanCaseScenarioMAE(:,1));
    signedRankStatistic(comparisonIndex) = stats.signedrank;

    stream = RandStream("mt19937ar","Seed", ...
        20262718+comparisonIndex);
    bootstrapDifference = nan(10000,1);
    for bootstrapIndex = 1:10000
        sampledIndex = randi(stream,numberOfCases,numberOfCases,1);
        bootstrapDifference(bootstrapIndex) = mean(difference(sampledIndex));
    end
    bootstrapLower(comparisonIndex) = prctile(bootstrapDifference,2.5);
    bootstrapUpper(comparisonIndex) = prctile(bootstrapDifference,97.5);
end

holmAdjustedP = holmAdjustment(rawP);
robustnessStatistics = table(scenarioNames(2:end)', ...
    repmat(numberOfCases,numberOfScenarios-1,1), ...
    meanPairedDifference,bootstrapLower,bootstrapUpper,rawP, ...
    holmAdjustedP,signedRankStatistic, ...
    'VariableNames',{'Scenario','IndependentCaseN','MeanMAEChange', ...
    'BootstrapCILower','BootstrapCIUpper','RawP','HolmAdjustedP', ...
    'SignedRankStatistic'});

%% 4. Figures
fig = figure("Color","w","Position",[100 100 1450 560]);
tiledlayout(1,2,"TileSpacing","compact","Padding","compact");

ax1 = nexttile;
histogram(ax1,permutationMAE,15,"FaceColor",[0.20 0.55 0.80]);
hold(ax1,"on");
xline(ax1,observedMAE,"r--","LineWidth",2, ...
    "Label",sprintf("Observed = %.4f",observedMAE));
hold(ax1,"off");
grid(ax1,"on");
xlabel(ax1,"MAE");
ylabel(ax1,"Permutation count");
title(ax1,"Case-wise elapsed-time permutation");

ax2 = nexttile;
errorbar(ax2,1:numberOfScenarios,MeanMAE,SDMAE,"o-", ...
    "LineWidth",1.8,"MarkerSize",7,"CapSize",8);
grid(ax2,"on");
xticks(ax2,1:numberOfScenarios);
xticklabels(ax2,scenarioNames);
xtickangle(ax2,20);
ylabel(ax2,"MAE (mean ± s.d. across perturbations)");
title(ax2,"Robustness to elapsed-time uncertainty");

for axisHandle = [ax1 ax2]
    if ~isempty(axisHandle.Toolbar)
        axisHandle.Toolbar.Visible = "off";
    end
    disableDefaultInteractivity(axisHandle);
end
drawnow;

%% 5. Save
writetable(permutationRepeats,fullfile(outputFolder, ...
    "time_permutation_repeats.csv"));
writetable(permutationSummary,fullfile(outputFolder, ...
    "time_permutation_summary.csv"));
writetable(robustnessRepeats,fullfile(outputFolder, ...
    "time_uncertainty_repeats.csv"));
writetable(scenarioSummary,fullfile(outputFolder, ...
    "time_uncertainty_summary.csv"));
writetable(robustnessStatistics,fullfile(outputFolder, ...
    "time_uncertainty_statistics.csv"));

save(fullfile(outputFolder,"time_shortcut_validation.mat"), ...
    "permutationRepeats","permutationSummary", ...
    "permutationPredictions","robustnessRepeats","scenarioSummary", ...
    "robustnessStatistics","caseScenarioMAE","observedPrediction");

exportgraphics(fig,fullfile(outputFolder, ...
    "time_shortcut_validation.png"),"Resolution",300);
exportgraphics(fig,fullfile(outputFolder, ...
    "time_shortcut_validation.pdf"),"ContentType","vector");

fprintf("\n========== Time-permutation summary ==========\n");
disp(permutationSummary);
fprintf("\n========== Time-uncertainty summary ==========\n");
disp(scenarioSummary);
fprintf("\nFiles saved to:\n%s\n",outputFolder);


function prediction = frozenPolicyPrediction( ...
    XProcess,XAllSensors,XOtherSensors,currentTime, ...
    saturationRatio,y,caseID,caseList,modelData)

prediction = nan(size(y));
for caseIndex = 1:numel(caseList)
    testMask = caseID == caseList(caseIndex);
    trainMask = ~testMask;
    XAll = [XProcess,XAllSensors,currentTime,currentTime.^2];
    XQuality = [XProcess,XOtherSensors, ...
        currentTime,currentTime.^2];
    predAll = fitRidgeAndPredict(XAll(trainMask,:),y(trainMask), ...
        XAll(testMask,:),modelData.bestLambdaTimeAll(caseIndex));
    predQuality = fitRidgeAndPredict(XQuality(trainMask,:),y(trainMask), ...
        XQuality(testMask,:),modelData.bestLambdaTimeQuality(caseIndex));
    predGated = predAll;
    switchMask = saturationRatio(testMask) >= ...
        modelData.bestGateThresholdTime(caseIndex);
    predGated(switchMask) = predQuality(switchMask);
    prediction(testMask) = predGated;
end
end


function caseMAE = caseMAEVector(y,prediction,caseID,caseList)
caseMAE = nan(numel(caseList),1);
for caseIndex = 1:numel(caseList)
    mask = caseID == caseList(caseIndex);
    caseMAE(caseIndex) = mean(abs(prediction(mask)-y(mask)));
end
end


function adjustedP = holmAdjustment(rawP)
numberOfTests = numel(rawP);
[sortedP,order] = sort(rawP);
adjustedSorted = nan(numberOfTests,1);
runningMaximum = 0;
for testIndex = 1:numberOfTests
    currentAdjusted = (numberOfTests-testIndex+1)*sortedP(testIndex);
    runningMaximum = max(runningMaximum,currentAdjusted);
    adjustedSorted(testIndex) = min(1,runningMaximum);
end
adjustedP = nan(numberOfTests,1);
adjustedP(order) = adjustedSorted;
end
