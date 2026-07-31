clear;
clc;
close all;

%% Reviewer-oriented high-wear underestimation and miss analysis
rootFolder = fileparts(mfilename("fullpath"));
inputFolder = fullfile(rootFolder,"additional_validation_results");
outputFolder = fullfile(rootFolder,"targeted_reviewer_experiments_20260719", ...
    "high_wear_underestimation");
if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

predictionFile = fullfile(inputFolder,"high_wear_predictions.csv");
assert(isfile(predictionFile), ...
    "Run step24_high_wear_safety_validation.m before this audit.");
T = readtable(predictionFile,"VariableNamingRule","preserve");

actual = T.ActualVB;
caseID = T.CaseID;
runID = T.RunID;
modelNames = ["PrimaryTimeGated";"SafetyWeightedTimeGated"];
predictionMatrix = [T.PrimaryPrediction,T.SafetyWeightedPrediction];
thresholds = [0.30 0.40 0.60 0.80];
largeUnderestimateMargin = 0.20;
caseList = unique(caseID,"stable");

%% Record-level descriptive metrics (no record-level inferential p values)
rows = cell(numel(modelNames)*numel(thresholds),17);
rowIndex = 0;
for modelIndex = 1:numel(modelNames)
    residual = predictionMatrix(:,modelIndex)-actual;
    for thresholdIndex = 1:numel(thresholds)
        threshold = thresholds(thresholdIndex);
        highMask = actual >= threshold;
        predictedHigh = predictionMatrix(:,modelIndex) >= threshold;
        highResidual = residual(highMask);
        underAmount = max(-highResidual,0);
        rowIndex = rowIndex+1;
        rows(rowIndex,:) = {modelNames(modelIndex),threshold,sum(highMask), ...
            mean(abs(highResidual)),sqrt(mean(highResidual.^2)), ...
            mean(highResidual),median(highResidual),mean(highResidual < 0), ...
            mean(underAmount),prctile(underAmount,90), ...
            sum(highResidual <= -largeUnderestimateMargin), ...
            mean(highResidual <= -largeUnderestimateMargin), ...
            sum(highMask & ~predictedHigh),mean(~predictedHigh(highMask)), ...
            mean(predictedHigh(highMask)), ...
            sum(actual >= 0.40 & predictionMatrix(:,modelIndex) < 0.20), ...
            largeUnderestimateMargin};
    end
end
thresholdMetrics = cell2table(rows,'VariableNames',cellstr( ...
    ["Model","ActualWearThreshold","HighWearRecordN","MAE","RMSE", ...
    "MeanResidual","MedianResidual","UnderpredictionRate", ...
    "MeanUnderpredictionAmount","P90UnderpredictionAmount", ...
    "LargeUnderpredictionN","LargeUnderpredictionRate", ...
    "AlertMissN","AlertMissRate","AlertSensitivity", ...
    "DangerousMissN_ActualGE04_PredLT02","LargeUnderestimateMargin"]));
thresholdMetrics.Model = string(thresholdMetrics.Model);

%% Case-level metrics: the independent analysis unit is the machining case
caseRows = cell(0,12);
for thresholdIndex = 1:numel(thresholds)
    threshold = thresholds(thresholdIndex);
    for caseIndex = 1:numel(caseList)
        caseMask = caseID == caseList(caseIndex);
        highMask = caseMask & actual >= threshold;
        if ~any(highMask)
            continue;
        end
        for modelIndex = 1:numel(modelNames)
            residual = predictionMatrix(highMask,modelIndex)-actual(highMask);
            predictedHigh = predictionMatrix(highMask,modelIndex) >= threshold;
            caseRows(end+1,:) = {caseList(caseIndex),threshold,sum(highMask), ...
                modelNames(modelIndex),mean(abs(residual)), ...
                sqrt(mean(residual.^2)),mean(residual), ...
                mean(residual < 0),mean(~predictedHigh),mean(predictedHigh), ...
                sum(residual <= -largeUnderestimateMargin), ...
                max(max(-residual,0))}; %#ok<SAGROW>
        end
    end
end
caseMetrics = cell2table(caseRows,'VariableNames',cellstr( ...
    ["CaseID","ActualWearThreshold","HighWearRecordN","Model","MAE", ...
    "RMSE","MeanResidual","UnderpredictionRate","AlertMissRate", ...
    "AlertSensitivity","LargeUnderpredictionN","MaximumUnderprediction"]));
caseMetrics.Model = string(caseMetrics.Model);

%% Last observed valid run in each case
lastRows = cell(numel(caseList)*numel(modelNames),10);
rowIndex = 0;
for caseIndex = 1:numel(caseList)
    mask = caseID == caseList(caseIndex);
    indices = find(mask);
    [~,localPosition] = max(runID(mask));
    index = indices(localPosition);
    for modelIndex = 1:numel(modelNames)
        rowIndex = rowIndex+1;
        residual = predictionMatrix(index,modelIndex)-actual(index);
        rowValues = {caseList(caseIndex),runID(index),actual(index), ...
            modelNames(modelIndex),predictionMatrix(index,modelIndex), ...
            residual,abs(residual),residual < 0, ...
            actual(index) >= 0.40 && predictionMatrix(index,modelIndex) < 0.40, ...
            actual(index) >= 0.40 && predictionMatrix(index,modelIndex) < 0.20};
        lastRows(rowIndex,:) = rowValues;
    end
end
lastRunMetrics = cell2table(lastRows,'VariableNames',cellstr( ...
    ["CaseID","LastRunID","ActualVB","Model","Prediction","Residual", ...
    "AbsoluteError","Underpredicted","MissedVB04Alert", ...
    "DangerousMissPredLT02"]));
lastRunMetrics.Model = string(lastRunMetrics.Model);

%% Paired case-level inference at the prespecified VB >= 0.40 threshold
primaryThreshold = 0.40;
primaryCase = caseMetrics(caseMetrics.ActualWearThreshold == primaryThreshold & ...
    caseMetrics.Model == "PrimaryTimeGated",:);
safetyCase = caseMetrics(caseMetrics.ActualWearThreshold == primaryThreshold & ...
    caseMetrics.Model == "SafetyWeightedTimeGated",:);
assert(isequal(primaryCase.CaseID,safetyCase.CaseID), ...
    "Case ordering mismatch in paired high-wear comparison.");

endpointNames = ["High-wear MAE";"Underprediction rate";"Alert miss rate"];
primaryValues = [primaryCase.MAE,primaryCase.UnderpredictionRate, ...
    primaryCase.AlertMissRate];
safetyValues = [safetyCase.MAE,safetyCase.UnderpredictionRate, ...
    safetyCase.AlertMissRate];
rawP = nan(numel(endpointNames),1);
signedRank = nan(numel(endpointNames),1);
meanDifference = nan(numel(endpointNames),1);
ciLower = nan(numel(endpointNames),1);
ciUpper = nan(numel(endpointNames),1);
bootstrapProbabilityImproves = nan(numel(endpointNames),1);
for endpointIndex = 1:numel(endpointNames)
    difference = safetyValues(:,endpointIndex)-primaryValues(:,endpointIndex);
    meanDifference(endpointIndex) = mean(difference);
    [rawP(endpointIndex),signedRank(endpointIndex)] = safeSignrank(difference);
    [ciLower(endpointIndex),ciUpper(endpointIndex), ...
        bootstrapProbabilityImproves(endpointIndex)] = ...
        bootstrapCaseDifference(difference,10000,20260719+endpointIndex);
end
holmAdjustedP = holmAdjust(rawP);
pairedStatistics = table(endpointNames,repmat(height(primaryCase),3,1), ...
    mean(primaryValues,1)',mean(safetyValues,1)',meanDifference, ...
    ciLower,ciUpper,rawP,holmAdjustedP,holmAdjustedP < 0.05, ...
    signedRank,bootstrapProbabilityImproves, ...
    'VariableNames',cellstr(["Endpoint","IndependentCaseN","PrimaryMean", ...
    "SafetyWeightedMean","MeanPairedDifference","BootstrapCILower", ...
    "BootstrapCIUpper","RawWilcoxonP","HolmAdjustedP", ...
    "SignificantAfterHolm","SignedRankStatistic", ...
    "BootstrapProbabilitySafetyImproves"]));

%% Worst underestimates and source-data table
sourceRows = table(T.OriginalIndex,caseID,runID,actual, ...
    predictionMatrix(:,1),predictionMatrix(:,2), ...
    predictionMatrix(:,1)-actual,predictionMatrix(:,2)-actual, ...
    'VariableNames',cellstr(["OriginalIndex","CaseID","RunID","ActualVB", ...
    "PrimaryPrediction","SafetyWeightedPrediction", ...
    "PrimaryResidual","SafetyWeightedResidual"]));
sourceRows.PrimaryUnderprediction = max(-sourceRows.PrimaryResidual,0);
sourceRows.SafetyWeightedUnderprediction = max(-sourceRows.SafetyWeightedResidual,0);
highSource = sourceRows(actual >= primaryThreshold,:);
worstUnderestimates = sortrows(highSource,"PrimaryUnderprediction","descend");
worstUnderestimates = worstUnderestimates(1:min(20,height(worstUnderestimates)),:);

%% Figure with transparent n and thresholds
fig = figure("Visible","off","Color","w","Position",[100 100 1600 1000]);
tl = tiledlayout(fig,2,2,"TileSpacing","compact","Padding","compact");
title(tl,"High-Wear Underestimation and Alert-Miss Audit", ...
    "FontWeight","bold","FontSize",15);
colors = [0.10 0.45 0.75;0.85 0.35 0.10];

ax1 = nexttile(tl);
hold(ax1,"on");
for modelIndex = 1:numel(modelNames)
    scatter(ax1,actual,predictionMatrix(:,modelIndex)-actual,26, ...
        colors(modelIndex,:),"filled","MarkerFaceAlpha",0.48);
end
yline(ax1,0,"k--","LineWidth",1.1);
xline(ax1,primaryThreshold,":","Color",[0.3 0.3 0.3],"LineWidth",1.2);
hold(ax1,"off"); grid(ax1,"on"); box(ax1,"on");
xlabel(ax1,"Actual VB"); ylabel(ax1,"Residual (prediction - actual)");
legend(ax1,replace(modelNames,"_"," "),"Location","southwest");
title(ax1,"Residuals across wear severity");

ax2 = nexttile(tl);
hold(ax2,"on");
for modelIndex = 1:numel(modelNames)
    mask = thresholdMetrics.Model == modelNames(modelIndex);
    sub = sortrows(thresholdMetrics(mask,:),"ActualWearThreshold");
    plot(ax2,sub.ActualWearThreshold,sub.AlertMissRate,"-o", ...
        "Color",colors(modelIndex,:),"MarkerFaceColor",colors(modelIndex,:), ...
        "LineWidth",1.7);
end
hold(ax2,"off"); grid(ax2,"on"); box(ax2,"on");
xlabel(ax2,"Actual-wear alert threshold"); ylabel(ax2,"Alert miss rate");
ylim(ax2,[0 1]);
legend(ax2,replace(modelNames,"_"," "),"Location","northwest");
title(ax2,"Threshold-dependent critical-wear misses");

ax3 = nexttile(tl);
primaryLast = lastRunMetrics.AbsoluteError(lastRunMetrics.Model == "PrimaryTimeGated");
safetyLast = lastRunMetrics.AbsoluteError(lastRunMetrics.Model == "SafetyWeightedTimeGated");
bar(ax3,[primaryLast safetyLast],"grouped");
grid(ax3,"on"); box(ax3,"on");
xticks(ax3,1:numel(caseList)); xticklabels(ax3,string(caseList));
xlabel(ax3,"Case"); ylabel(ax3,"Absolute error at last valid run");
legend(ax3,{"Primary","Safety weighted"},"Location","northwest");
title(ax3,"End-of-sequence error (n = 16 cases)");

ax4 = nexttile(tl);
hold(ax4,"on");
for caseIndex = 1:height(primaryCase)
    if safetyCase.MAE(caseIndex) <= primaryCase.MAE(caseIndex)
        lineColor = [0.10 0.55 0.35];
    else
        lineColor = [0.80 0.25 0.15];
    end
    plot(ax4,[1 2],[primaryCase.MAE(caseIndex),safetyCase.MAE(caseIndex)], ...
        "-o","Color",lineColor,"MarkerFaceColor",lineColor,"LineWidth",1.2);
end
hold(ax4,"off"); grid(ax4,"on"); box(ax4,"on");
xlim(ax4,[0.7 2.3]); xticks(ax4,[1 2]);
xticklabels(ax4,{"Primary","Safety weighted"});
ylabel(ax4,"Case-level MAE for VB >= 0.40");
title(ax4,sprintf("Paired high-wear cases (n = %d)",height(primaryCase)));

exportgraphics(fig,fullfile(outputFolder,"high_wear_underestimation_audit.png"), ...
    "Resolution",300);
exportgraphics(fig,fullfile(outputFolder,"high_wear_underestimation_audit.pdf"), ...
    "ContentType","vector");
close(fig);

%% Save
writetable(thresholdMetrics,fullfile(outputFolder,"high_wear_threshold_metrics.csv"));
writetable(caseMetrics,fullfile(outputFolder,"high_wear_case_level_metrics.csv"));
writetable(lastRunMetrics,fullfile(outputFolder,"last_valid_run_metrics.csv"));
writetable(pairedStatistics,fullfile(outputFolder,"high_wear_paired_statistics.csv"));
writetable(sourceRows,fullfile(outputFolder,"high_wear_prediction_source_data.csv"));
writetable(worstUnderestimates,fullfile(outputFolder,"worst_high_wear_underestimates.csv"));
save(fullfile(outputFolder,"high_wear_underestimation_audit.mat"), ...
    "thresholdMetrics","caseMetrics","lastRunMetrics","pairedStatistics", ...
    "sourceRows","worstUnderestimates","thresholds", ...
    "largeUnderestimateMargin","primaryThreshold");
writeReadme(outputFolder,thresholdMetrics,pairedStatistics, ...
    numel(caseList),height(primaryCase),largeUnderestimateMargin);

fprintf("========== High-wear threshold metrics ==========\n");
disp(thresholdMetrics);
fprintf("========== Paired independent-case statistics ==========\n");
disp(pairedStatistics);
fprintf("Results saved to:\n%s\n",outputFolder);


function [p,statistic] = safeSignrank(delta)
delta = delta(isfinite(delta) & delta ~= 0);
if isempty(delta)
    p = 1;
    statistic = 0;
    return;
end
try
    [p,~,stats] = signrank(delta,0,"method","exact");
catch
    [p,~,stats] = signrank(delta,0);
end
if isfield(stats,"signedrank")
    statistic = stats.signedrank;
else
    statistic = NaN;
end
end


function adjusted = holmAdjust(p)
p = p(:);
numberOfTests = numel(p);
[sortedP,order] = sort(p);
adjustedSorted = nan(numberOfTests,1);
for index = 1:numberOfTests
    adjustedSorted(index) = min(1,(numberOfTests-index+1)*sortedP(index));
end
for index = 2:numberOfTests
    adjustedSorted(index) = max(adjustedSorted(index),adjustedSorted(index-1));
end
adjusted = nan(numberOfTests,1);
adjusted(order) = adjustedSorted;
end


function [lower,upper,probability] = bootstrapCaseDifference(delta,nBootstrap,seed)
stream = RandStream("mt19937ar","Seed",seed);
numberOfCases = numel(delta);
bootstrapMean = nan(nBootstrap,1);
for bootstrapIndex = 1:nBootstrap
    index = randi(stream,numberOfCases,numberOfCases,1);
    bootstrapMean(bootstrapIndex) = mean(delta(index));
end
limits = prctile(bootstrapMean,[2.5 97.5]);
lower = limits(1);
upper = limits(2);
probability = mean(bootstrapMean < 0);
end


function writeReadme(outputFolder,thresholdMetrics,pairedStatistics, ...
    totalCaseN,highWearCaseN,largeMargin)
fileID = fopen(fullfile(outputFolder,"README_HIGH_WEAR_AUDIT.txt"), ...
    "w","n","UTF-8");
assert(fileID > 0,"Could not create high-wear README.");
cleanup = onCleanup(@() fclose(fileID)); %#ok<NASGU>
fprintf(fileID,"HIGH-WEAR UNDERESTIMATION AND ALERT-MISS AUDIT\n");
fprintf(fileID,"Generated with MATLAB %s\n\n",version);
fprintf(fileID,"Analysis units\n");
fprintf(fileID,"- Total machining cases: n=%d.\n",totalCaseN);
fprintf(fileID,"- Cases containing VB >= 0.40: n=%d.\n",highWearCaseN);
fprintf(fileID,"- Run-level rows are descriptive observations, not independent inferential replicates.\n");
fprintf(fileID,"- Wilcoxon tests and bootstrap confidence intervals use paired case-level summaries.\n");
fprintf(fileID,"- Holm adjustment covers the three prespecified case-level endpoints.\n\n");
fprintf(fileID,"Definitions\n");
fprintf(fileID,"- Residual = prediction - actual; negative values denote underestimation.\n");
fprintf(fileID,"- Alert miss = actual wear at or above a threshold but prediction below that same threshold.\n");
fprintf(fileID,"- Large underestimation is a descriptive margin of %.2f VB; it is not asserted to be a universal industrial safety limit.\n",largeMargin);
fprintf(fileID,"- Dangerous miss count is reported descriptively as actual VB >= 0.40 with predicted VB < 0.20.\n\n");
fprintf(fileID,"Record-level descriptive results\n");
for rowIndex = 1:height(thresholdMetrics)
    fprintf(fileID,"%s, actual VB >= %.2f: n=%d, MAE=%.6f, mean residual=%.6f, underprediction rate=%.4f, alert miss rate=%.4f, large-underprediction n=%d.\n", ...
        thresholdMetrics.Model(rowIndex), ...
        thresholdMetrics.ActualWearThreshold(rowIndex), ...
        thresholdMetrics.HighWearRecordN(rowIndex), ...
        thresholdMetrics.MAE(rowIndex), ...
        thresholdMetrics.MeanResidual(rowIndex), ...
        thresholdMetrics.UnderpredictionRate(rowIndex), ...
        thresholdMetrics.AlertMissRate(rowIndex), ...
        thresholdMetrics.LargeUnderpredictionN(rowIndex));
end
fprintf(fileID,"\nPaired case-level inference for actual VB >= 0.40\n");
for rowIndex = 1:height(pairedStatistics)
    fprintf(fileID,"%s: n=%d, primary=%.6f, safety=%.6f, difference=%.6f, bootstrap 95%% CI=[%.6f, %.6f], raw p=%.6g, Holm p=%.6g.\n", ...
        pairedStatistics.Endpoint(rowIndex), ...
        pairedStatistics.IndependentCaseN(rowIndex), ...
        pairedStatistics.PrimaryMean(rowIndex), ...
        pairedStatistics.SafetyWeightedMean(rowIndex), ...
        pairedStatistics.MeanPairedDifference(rowIndex), ...
        pairedStatistics.BootstrapCILower(rowIndex), ...
        pairedStatistics.BootstrapCIUpper(rowIndex), ...
        pairedStatistics.RawWilcoxonP(rowIndex), ...
        pairedStatistics.HolmAdjustedP(rowIndex));
end
fprintf(fileID,"\nInterpretation boundary\n");
fprintf(fileID,"- The safety-weighted model is a secondary analysis and does not replace the prespecified primary model.\n");
fprintf(fileID,"- Any improvement in miss rate must be interpreted together with overall MAE and false-alert trade-offs.\n");
end
