paperRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
clearvars -except paperRoot;
clc;

%% Controlled nominal-condition-overlap sensitivity analysis
% NASA contains eight nominal cutting conditions with two independent cases
% per condition. Each split below contains eight training cases and eight test
% cases. Only the proportion of nominal conditions represented on both sides
% is varied (0%, 25%, 50%, 75% or 100%). Thus, training/test case counts are
% fixed while same-condition support changes. Repeated partitions are Monte
% Carlo protocol samples and are not treated as independent inferential units.

addpath(paperRoot);
resultDir = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(resultDir, 'dir'); mkdir(resultDir); end

S = load(fullfile(paperRoot, 'mill', 'results', 'quality_aware_with_time.mat'));
T = S.timeAwareTable;

y = T.VB;
caseID = T.CaseID;
elapsedTime = T.ElapsedTime;
qualityRatio = T.smcDC_SaturationRatio;
[conditionID, conditionKey] = findgroups(T(:, {'Material','Feed','DOC'}));
conditionList = unique(conditionID, 'stable');
caseList = unique(caseID, 'stable');

assert(numel(conditionList) == 8, 'Expected eight nominal conditions.');
assert(numel(caseList) == 16, 'Expected sixteen independent cases.');

casesByCondition = cell(numel(conditionList), 1);
for k = 1:numel(conditionList)
    casesByCondition{k} = unique(caseID(conditionID == conditionList(k)), 'stable');
    assert(numel(casesByCondition{k}) == 2, ...
        'Each nominal condition must contain exactly two independent cases.');
end

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
overlapConditionN = [0 2 4 6 8];
overlapPercent = 100 * overlapConditionN / numel(conditionList);
repeatN = 100;
baseSeed = 2026081400;

rows = cell(numel(overlapConditionN) * repeatN, 20);
row = 0;
for levelIndex = 1:numel(overlapConditionN)
    nSplit = overlapConditionN(levelIndex);
    nFullTest = (numel(conditionList) - nSplit) / 2;
    assert(mod(nFullTest, 1) == 0, 'Invalid overlap construction.');

    for repeatIndex = 1:repeatN
        seed = baseSeed + 1000 * levelIndex + repeatIndex;
        rng(seed, 'twister');

        order = randperm(numel(conditionList));
        splitConditions = conditionList(order(1:nSplit));
        remainingConditions = conditionList(order(nSplit+1:end));
        fullTestConditions = remainingConditions(1:nFullTest);
        fullTrainConditions = remainingConditions(nFullTest+1:end);

        testCases = [];
        trainCases = [];
        for k = 1:numel(splitConditions)
            pair = casesByCondition{conditionList == splitConditions(k)};
            if rand < 0.5
                testCases(end+1,1) = pair(1); %#ok<SAGROW>
                trainCases(end+1,1) = pair(2); %#ok<SAGROW>
            else
                testCases(end+1,1) = pair(2); %#ok<SAGROW>
                trainCases(end+1,1) = pair(1); %#ok<SAGROW>
            end
        end
        for k = 1:numel(fullTestConditions)
            pair = casesByCondition{conditionList == fullTestConditions(k)};
            testCases = [testCases; pair(:)]; %#ok<AGROW>
        end
        for k = 1:numel(fullTrainConditions)
            pair = casesByCondition{conditionList == fullTrainConditions(k)};
            trainCases = [trainCases; pair(:)]; %#ok<AGROW>
        end

        assert(numel(testCases) == 8 && numel(trainCases) == 8, ...
            'Every split must contain eight training and eight test cases.');
        assert(isempty(intersect(testCases, trainCases)), ...
            'Training and test cases must be disjoint.');

        testMask = ismember(caseID, testCases);
        trainMask = ismember(caseID, trainCases);
        assert(all(trainMask | testMask), 'Every record must belong to train or test.');

        [predAll, predReduced, predSQTR, lambdaAll, lambdaReduced, gate] = ...
            groupedGatedRidgePrediction(XAllTime, XReducedTime, qualityRatio, ...
            y, caseID, trainMask, testMask, lambdaGrid, gateGrid);

        testY = y(testMask);
        fixedPooled = mean(abs(predAll - testY));
        reducedPooled = mean(abs(predReduced - testY));
        sqtrPooled = mean(abs(predSQTR - testY));

        testCaseList = unique(caseID(testMask), 'stable');
        fixedCaseMAE = nan(numel(testCaseList), 1);
        reducedCaseMAE = nan(numel(testCaseList), 1);
        sqtrCaseMAE = nan(numel(testCaseList), 1);
        testCaseID = caseID(testMask);
        for c = 1:numel(testCaseList)
            mask = testCaseID == testCaseList(c);
            fixedCaseMAE(c) = mean(abs(predAll(mask) - testY(mask)));
            reducedCaseMAE(c) = mean(abs(predReduced(mask) - testY(mask)));
            sqtrCaseMAE(c) = mean(abs(predSQTR(mask) - testY(mask)));
        end

        row = row + 1;
        rows(row,:) = {overlapPercent(levelIndex), nSplit, repeatIndex, seed, ...
            8, 8, sum(trainMask), sum(testMask), ...
            fixedPooled, reducedPooled, sqtrPooled, ...
            mean(fixedCaseMAE), mean(reducedCaseMAE), mean(sqtrCaseMAE), ...
            fixedPooled - sqtrPooled, mean(fixedCaseMAE - sqtrCaseMAE), ...
            lambdaAll, lambdaReduced, gate, ...
            strjoin(string(sort(testCases(:)')), '|')};
    end
end

partitionTable = cell2table(rows, 'VariableNames', ...
    {'ConditionOverlapPercent','OverlappingConditionN','Repeat','Seed', ...
    'TrainCaseN','TestCaseN','TrainRecordN','TestRecordN', ...
    'FixedPooledMAE','ReducedPooledMAE','SQTRPooledMAE', ...
    'FixedCaseMacroMAE','ReducedCaseMacroMAE','SQTRCaseMacroMAE', ...
    'FixedMinusSQTRPooled','FixedMinusSQTRCaseMacro', ...
    'LambdaAll','LambdaReduced','Gate','TestCases'});

summaryRows = cell(numel(overlapPercent), 17);
for levelIndex = 1:numel(overlapPercent)
    mask = partitionTable.ConditionOverlapPercent == overlapPercent(levelIndex);
    f = partitionTable.FixedPooledMAE(mask);
    s = partitionTable.SQTRPooledMAE(mask);
    fm = partitionTable.FixedCaseMacroMAE(mask);
    sm = partitionTable.SQTRCaseMacroMAE(mask);
    dp = partitionTable.FixedMinusSQTRPooled(mask);
    dm = partitionTable.FixedMinusSQTRCaseMacro(mask);
    summaryRows(levelIndex,:) = {overlapPercent(levelIndex), ...
        overlapConditionN(levelIndex), repeatN, ...
        mean(f), std(f), prctile(f,2.5), prctile(f,97.5), ...
        mean(s), std(s), prctile(s,2.5), prctile(s,97.5), ...
        mean(fm), mean(sm), mean(dp), prctile(dp,2.5), ...
        prctile(dp,97.5), mean(dm)};
end
summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'ConditionOverlapPercent','OverlappingConditionN','MonteCarloRepeatN', ...
    'FixedPooledMean','FixedPooledSD','FixedPooledP025','FixedPooledP975', ...
    'SQTRPooledMean','SQTRPooledSD','SQTRPooledP025','SQTRPooledP975', ...
    'FixedCaseMacroMean','SQTRCaseMacroMean','FixedMinusSQTRPooledMean', ...
    'FixedMinusSQTRPooledP025','FixedMinusSQTRPooledP975', ...
    'FixedMinusSQTRCaseMacroMean'});

writetable(partitionTable, fullfile(resultDir, 'protocol_overlap_partitions.csv'));
writetable(summaryTable, fullfile(resultDir, 'protocol_overlap_summary.csv'));
save(fullfile(resultDir, 'protocol_overlap_results.mat'));

disp(summaryTable);

