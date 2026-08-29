paperRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
clearvars -except paperRoot;
clc;

%% Strategy-level paired inference and denominator sensitivity
% The 30 fault randomisations have already been averaged within each
% independent NUAA experiment or PHM tool. This analysis therefore uses
% the 12 independent-unit rows, not the technical repetitions, for paired
% inference. Three quality-conditioned responses are compared directly
% with fixed fusion in one prespecified Holm family.

resultDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(resultDir, 'dir'); mkdir(resultDir); end

sourceFile = string(getenv('SQTR_STRATEGY_SOURCE_FILE'));
if strlength(sourceFile)==0
    sourceFile = fullfile(paperRoot, 'additional_validation_results', ...
        'reviewer_improvements', 'random_fault_experiment_means.csv');
end
outputOverride = string(getenv('SQTR_STRATEGY_RESULT_DIR'));
if strlength(outputOverride)>0
    resultDir = outputOverride;
    if ~exist(resultDir, 'dir'); mkdir(resultDir); end
end
T = readtable(sourceFile, 'VariableNamingRule', 'preserve');

methodNames = ["Median replacement";"Mask-aware dropout";"SQTR"];
relativeColumns = {'MedianReplacementRelativeMAE', ...
    'MaskDropoutRelativeMAE','AutoRelativeMAE'};
cleanColumns = {'CleanMedianReplacementMAE', ...
    'CleanMaskDropoutMAE','CleanAutoMAE'};

fixedRelative = T.FullRelativeMAE;
rawP = nan(numel(methodNames),1);
comparisonRows = cell(numel(methodNames),11);
rng(2026081401,'twister');
B = 10000;
for m = 1:numel(methodNames)
    responseRelative = T.(relativeColumns{m});
    difference = fixedRelative - responseRelative; % positive favours response
    [rawP(m),~,stats] = signrank(fixedRelative,responseRelative, ...
        'tail','both','method','exact');
    boot = nan(B,1); n = numel(difference);
    for b = 1:B
        idx = randi(n,n,1);
        boot(b) = mean(difference(idx));
    end
    ci = prctile(boot,[2.5 97.5]);
    comparisonRows(m,:) = {methodNames(m),n,mean(fixedRelative), ...
        mean(responseRelative),mean(difference),ci(1),ci(2), ...
        sum(difference>0),sum(difference<0),sum(difference==0),stats.signedrank};
end
holmP = holmAdjust(rawP);
strategyComparison = cell2table(comparisonRows,'VariableNames', ...
    {'Response','IndependentUnitN','FixedRelativeMAE','ResponseRelativeMAE', ...
    'FixedMinusResponse','BootstrapCI_Lower','BootstrapCI_Upper', ...
    'UnitsFavourResponse','UnitsFavourFixed','TiedUnits','SignedRankStatistic'});
strategyComparison.RawP = rawP;
strategyComparison.HolmAdjustedP = holmP;
strategyComparison.SignificantAfterHolm = holmP < 0.05;

%% Method-specific clean denominator sensitivity
denominatorRows = cell(height(T)*4,9); row = 0;
allMethodNames = ["Fixed fusion";methodNames];
allRelativeColumns = [{'FullRelativeMAE'},relativeColumns];
allCleanColumns = [{'CleanFullMAE'},cleanColumns];
for i = 1:height(T)
    fixedClean = T.CleanFullMAE(i);
    for m = 1:numel(allMethodNames)
        fixedCleanRelative = T.(allRelativeColumns{m})(i);
        degradedMAE = fixedCleanRelative * fixedClean;
        ownClean = T.(allCleanColumns{m})(i);
        ownCleanRelative = degradedMAE / ownClean;
        row = row + 1;
        denominatorRows(row,:) = {string(T.Dataset(i)),string(T.Experiment(i)), ...
            allMethodNames(m),fixedClean,ownClean,degradedMAE, ...
            fixedCleanRelative,ownCleanRelative,ownCleanRelative-fixedCleanRelative};
    end
end
denominatorTable = cell2table(denominatorRows,'VariableNames', ...
    {'Dataset','Experiment','Method','FixedCleanMAE','OwnCleanMAE', ...
    'DegradedMAE','FixedCleanNormalisedMAE','OwnCleanNormalisedMAE', ...
    'NormalisationDifference'});

summaryRows = cell(2*numel(allMethodNames),8); row = 0;
datasets = ["NUAA","PHM2010"];
for d = 1:numel(datasets)
    for m = 1:numel(allMethodNames)
        mask = denominatorTable.Dataset==datasets(d) & ...
            denominatorTable.Method==allMethodNames(m);
        row = row + 1;
        summaryRows(row,:) = {datasets(d),allMethodNames(m),sum(mask), ...
            mean(denominatorTable.FixedCleanNormalisedMAE(mask)), ...
            mean(denominatorTable.OwnCleanNormalisedMAE(mask)), ...
            mean(denominatorTable.NormalisationDifference(mask)), ...
            min(denominatorTable.NormalisationDifference(mask)), ...
            max(denominatorTable.NormalisationDifference(mask))};
    end
end
denominatorSummary = cell2table(summaryRows,'VariableNames', ...
    {'Dataset','Method','IndependentUnitN','MeanFixedCleanNormalisedMAE', ...
    'MeanOwnCleanNormalisedMAE','MeanNormalisationDifference', ...
    'MinimumNormalisationDifference','MaximumNormalisationDifference'});

writetable(strategyComparison,fullfile(resultDir,'strategy_vs_fixed_holm_family.csv'));
writetable(denominatorTable,fullfile(resultDir,'own_clean_denominator_units.csv'));
writetable(denominatorSummary,fullfile(resultDir,'own_clean_denominator_summary.csv'));
save(fullfile(resultDir,'strategy_family_and_denominator.mat'));

disp(strategyComparison);
disp(denominatorSummary);

function adjusted=holmAdjust(p)
    [sortedP,order]=sort(p(:)); m=numel(sortedP); sortedAdjusted=zeros(m,1);
    running=0;
    for i=1:m
        running=max(running,(m-i+1)*sortedP(i));
        sortedAdjusted(i)=min(1,running);
    end
    adjusted=zeros(m,1); adjusted(order)=sortedAdjusted;
end
