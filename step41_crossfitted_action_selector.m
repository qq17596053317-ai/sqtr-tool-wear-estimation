clear; clc; close all;

paperRoot=fileparts(mfilename('fullpath'));
dataRoot=string(getenv('SQTR_RAW_DATA_DIR'));
if strlength(dataRoot)==0, dataRoot="C:\Users\66485\Desktop\论文"; end
resultDir=string(getenv('SQTR_ACTION_SELECTOR_RESULT_DIR'));
if strlength(resultDir)==0
    resultDir=fullfile(paperRoot,'targeted_reviewer_experiments', ...
        'crossfitted_action_selector','results');
end
if ~exist(resultDir,'dir'), mkdir(resultDir); end

trainRepeatCount=str2double(getenv('SQTR_ACTION_TRAIN_REPEATS'));
if ~isfinite(trainRepeatCount), trainRepeatCount=5; end
testRepeatCount=str2double(getenv('SQTR_ACTION_TEST_REPEATS'));
if ~isfinite(testRepeatCount), testRepeatCount=30; end
runTransferAudits=any(strcmpi(strtrim(getenv('SQTR_ACTION_TRANSFER_AUDITS')), ...
    {'1','true','yes'}));
bootstrapCount=10000;
treeMaxSplits=3;
baseSeed=2026081500;
testSeedBase=2026085000;
faultTypes=["Saturation","Dropout","Bias","Noise", ...
    "SoftClipping","IntermittentDropout","GradualDrift", ...
    "HeteroscedasticNoise"];
actionNames=["Full","MedianReplacement","MaskDropout","Exclusion"];

S=load(fullfile(paperRoot,'external_validation_results', ...
    'external_validation_complete_results.mat'),'nuaa','phm','lambdaGrid');
dropoutRates=[0.25 0.50 1.00]; detectorFloor=4; detectorQuantile=99;

fprintf('Preparing outer grouped policies...\n');
nuaa=prepareActionData(S.nuaa, ...
    fullfile(dataRoot,'nuaa_orthogonal_bundle_high_resolution.csv'), ...
    "NUAA",["force_z","vibration1","vibration2"], ...
    ["force_z","vibration_x","vibration_y"],S.lambdaGrid, ...
    dropoutRates,baseSeed,detectorFloor,detectorQuantile);
phm=prepareActionData(S.phm, ...
    fullfile(dataRoot,'phm2010_bundle_high_resolution.csv'), ...
    "PHM2010",["force_z","vibration_x","vibration_y"], ...
    ["force_z","vibration_x","vibration_y"],S.lambdaGrid, ...
    dropoutRates,baseSeed+50000,detectorFloor,detectorQuantile);
prepared={nuaa,phm};

unitRows={}; unitRow=0; predictionTables=cell(12,1);
transferRows={}; transferRow=0; crossDatasetRows={}; crossDatasetRow=0;
totalTimer=tic;
for testDatasetIndex=1:2
    Ptest=prepared{testDatasetIndex};
    testUnits=unique(Ptest.D.experiment,'stable');
    for testUnitIndex=1:numel(testUnits)
        testUnit=testUnits(testUnitIndex);
        fprintf('Outer action-policy unit %s | %s (%d/%d)\n', ...
            Ptest.name,testUnit,testUnitIndex,numel(testUnits));
        trainingParts=cell(0,1); part=0;
        for sourceDatasetIndex=1:2
            Psource=prepared{sourceDatasetIndex};
            sourceUnits=unique(Psource.D.experiment,'stable');
            for sourceUnitIndex=1:numel(sourceUnits)
                sourceUnit=sourceUnits(sourceUnitIndex);
                if sourceDatasetIndex==testDatasetIndex && sourceUnit==testUnit
                    continue;
                end
                if sourceDatasetIndex==testDatasetIndex
                    trainingMask=Psource.D.experiment~=testUnit & ...
                        Psource.D.experiment~=sourceUnit;
                    hyperPolicy=Ptest.outerPolicies{testUnitIndex};
                    fixedSeed=baseSeed+100000*testDatasetIndex+ ...
                        1000*testUnitIndex+sourceUnitIndex;
                    sourcePolicy=fitFixedPolicy(Psource.D,Psource.cleanQuality, ...
                        Psource.featureBlocks,trainingMask,hyperPolicy,fixedSeed, ...
                        detectorFloor,detectorQuantile);
                else
                    sourcePolicy=Psource.outerPolicies{sourceUnitIndex};
                end
                part=part+1;
                trainingParts{part,1}=makeActionRows(Psource,sourceUnit, ...
                    sourcePolicy,faultTypes,trainRepeatCount,baseSeed,true);
            end
        end
        trainingTable=vertcat(trainingParts{:});
        trainingTable.RowWeight=balancedUnitFamilyWeights(trainingTable);
        actionModels=fitActionModels(trainingTable,treeMaxSplits);
        bestFixedIndex=selectBestFixedAction(trainingTable);
        selectiveThreshold=selectSelectiveThreshold(trainingTable, ...
            bestFixedIndex,treeMaxSplits);

        testPolicy=Ptest.outerPolicies{testUnitIndex};
        testTable=makeActionRows(Ptest,testUnit,testPolicy,faultTypes, ...
            testRepeatCount,testSeedBase,true);
        [selectedIndex,predictedLoss]=selectActions(actionModels,testTable);
        actualLoss=lossMatrix(testTable);
        selectedLoss=actualLoss(sub2ind(size(actualLoss), ...
            (1:height(testTable))',selectedIndex));
        [oracleLoss,oracleIndex]=min(actualLoss,[],2);
        [selectiveIndex,selectiveGain]=selectSelectiveActions(predictedLoss, ...
            testTable,bestFixedIndex,selectiveThreshold);
        selectiveLoss=actualLoss(sub2ind(size(actualLoss), ...
            (1:height(testTable))',selectiveIndex));
        isDegraded=testTable.FaultType~="Clean";
        isClean=~isDegraded;
        actionRelative=mean(actualLoss(isDegraded,:),1);
        learnedRelative=mean(selectedLoss(isDegraded));
        selectiveRelative=mean(selectiveLoss(isDegraded));
        oracleRelative=mean(oracleLoss(isDegraded));
        bestFixedRelative=actionRelative(bestFixedIndex);
        cleanNonFullRate=mean(selectedIndex(isClean)~=1);
        degradedNonFullRate=mean(selectedIndex(isDegraded)~=1);
        degradedRegret=mean(selectedLoss(isDegraded)-oracleLoss(isDegraded));
        selectiveRegret=mean(selectiveLoss(isDegraded)-oracleLoss(isDegraded));
        selectiveOverrideRate=mean(selectiveIndex(isDegraded)~=bestFixedIndex);
        cleanOverrideRate=mean(selectiveIndex(isClean)~=bestFixedIndex);
        selectionFractions=arrayfun(@(a)mean(selectedIndex(isDegraded)==a),1:4);
        unitRow=unitRow+1;
        unitRows(unitRow,:)={Ptest.name,testUnit,sum(isDegraded), ...
            trainRepeatCount,testRepeatCount,bestFixedIndex, ...
            actionNames(bestFixedIndex),selectiveThreshold, ...
            actionRelative(1),actionRelative(2), ...
            actionRelative(3),actionRelative(4),bestFixedRelative, ...
            learnedRelative,selectiveRelative,oracleRelative,degradedRegret, ...
            selectiveRegret,selectiveOverrideRate,cleanOverrideRate,cleanNonFullRate, ...
            degradedNonFullRate,selectionFractions(1),selectionFractions(2), ...
            selectionFractions(3),selectionFractions(4)};

        testTable.SelectedAction=actionNames(selectedIndex)';
        testTable.SelectiveAction=actionNames(selectiveIndex)';
        testTable.OracleAction=actionNames(oracleIndex)';
        testTable.SelectedRelativeAbsoluteError=selectedLoss;
        testTable.SelectiveRelativeAbsoluteError=selectiveLoss;
        testTable.SelectivePredictedGain=selectiveGain;
        testTable.OracleRelativeAbsoluteError=oracleLoss;
        testTable.PredictedFullLoss=predictedLoss(:,1);
        testTable.PredictedMedianLoss=predictedLoss(:,2);
        testTable.PredictedMaskLoss=predictedLoss(:,3);
        testTable.PredictedExclusionLoss=predictedLoss(:,4);
        predictionTables{unitRow}=testTable;

        if runTransferAudits
            for heldoutIndex=1:numel(faultTypes)
                heldoutFault=faultTypes(heldoutIndex);
                transferTraining=trainingTable(trainingTable.FaultType~=heldoutFault,:);
                transferTraining.RowWeight=balancedUnitFamilyWeights(transferTraining);
                transferModels=fitActionModels(transferTraining,treeMaxSplits);
                transferFixedIndex=selectBestFixedAction(transferTraining);
                familyMask=testTable.FaultType==heldoutFault;
                familyTable=testTable(familyMask,:);
                [familySelected,~]=selectActions(transferModels,familyTable);
                familyActual=lossMatrix(familyTable);
                familyLearned=familyActual(sub2ind(size(familyActual), ...
                    (1:height(familyTable))',familySelected));
                familyOracle=min(familyActual,[],2);
                transferRow=transferRow+1;
                transferRows(transferRow,:)={Ptest.name,testUnit,heldoutFault, ...
                    height(familyTable),actionNames(transferFixedIndex), ...
                    mean(familyActual(:,transferFixedIndex)),mean(familyLearned), ...
                    mean(familyOracle),mean(familySelected~=transferFixedIndex)};
            end

            crossTraining=trainingTable(trainingTable.Dataset~=Ptest.name,:);
            crossTraining.RowWeight=balancedUnitFamilyWeights(crossTraining);
            crossModels=fitActionModels(crossTraining,treeMaxSplits);
            crossFixedIndex=selectBestFixedAction(crossTraining);
            degradedTable=testTable(isDegraded,:);
            [crossSelected,~]=selectActions(crossModels,degradedTable);
            crossActual=lossMatrix(degradedTable);
            crossLearned=crossActual(sub2ind(size(crossActual), ...
                (1:height(degradedTable))',crossSelected));
            crossOracle=min(crossActual,[],2);
            crossDatasetRow=crossDatasetRow+1;
            sourceName=prepared{3-testDatasetIndex}.name;
            crossDatasetRows(crossDatasetRow,:)={Ptest.name,testUnit,sourceName, ...
                height(degradedTable),actionNames(crossFixedIndex), ...
                mean(crossActual(:,crossFixedIndex)),mean(crossLearned), ...
                mean(crossOracle),mean(crossSelected~=crossFixedIndex)};
        end
    end
end
runtimeSeconds=toc(totalTimer);

unitTable=cell2table(unitRows,'VariableNames', ...
    {'Dataset','Unit','DegradedRecordScenarioN','TrainingTechnicalRepeats', ...
    'TestTechnicalRepeats','TrainingSelectedFixedIndex', ...
    'TrainingSelectedFixedAction','SelectiveGainThreshold', ...
    'FullRelativeMAE','MedianRelativeMAE', ...
    'MaskDropoutRelativeMAE','ExclusionRelativeMAE', ...
    'TrainingSelectedFixedRelativeMAE','LearnedPolicyRelativeMAE', ...
    'SelectivePolicyRelativeMAE','OracleEnvelopeRelativeMAE', ...
    'MeanRegretToOracle','SelectiveMeanRegretToOracle', ...
    'SelectiveOverrideRate','CleanSelectiveOverrideRate','CleanNonFullActionRate', ...
    'DegradedNonFullActionRate','SelectedFullFraction', ...
    'SelectedMedianFraction','SelectedMaskFraction', ...
    'SelectedExclusionFraction'});
unitTable.Dataset=string(unitTable.Dataset); unitTable.Unit=string(unitTable.Unit);
unitTable.TrainingSelectedFixedAction=string(unitTable.TrainingSelectedFixedAction);
predictionTable=vertcat(predictionTables{:});

primaryDifference=unitTable.TrainingSelectedFixedRelativeMAE- ...
    unitTable.SelectivePolicyRelativeMAE;
[primaryLower,primaryUpper]=clusterBootstrapMeanCI(primaryDifference, ...
    bootstrapCount,2026081591);
[primaryP,primaryW]=exactSignedRank(primaryDifference);
primaryTable=table(height(unitTable),trainRepeatCount,testRepeatCount, ...
    mean(unitTable.TrainingSelectedFixedRelativeMAE), ...
    mean(unitTable.SelectivePolicyRelativeMAE),mean(primaryDifference), ...
    primaryLower,primaryUpper,sum(primaryDifference>0), ...
    sum(primaryDifference<0),sum(primaryDifference==0),primaryW,primaryP, ...
    'VariableNames',{'IndependentUnitN','TrainingTechnicalRepeats', ...
    'TestTechnicalRepeats','TrainingSelectedFixedRelativeMAE', ...
    'SelectivePolicyRelativeMAE','FixedMinusSelective','BootstrapCI_Lower', ...
    'BootstrapCI_Upper','UnitsFavourSelective','UnitsFavourFixed','TiedUnits', ...
    'SignedRankStatistic','ExactP'});

comparisonValues=[unitTable.FullRelativeMAE,unitTable.MedianRelativeMAE, ...
    unitTable.MaskDropoutRelativeMAE,unitTable.ExclusionRelativeMAE];
comparisonRows=cell(4,13); rawP=nan(4,1);
for actionIndex=1:4
    difference=comparisonValues(:,actionIndex)-unitTable.SelectivePolicyRelativeMAE;
    [ciLower,ciUpper]=clusterBootstrapMeanCI(difference,bootstrapCount, ...
        2026081600+actionIndex);
    [rawP(actionIndex),W]=exactSignedRank(difference);
    comparisonRows(actionIndex,:)={actionNames(actionIndex),height(unitTable), ...
        mean(comparisonValues(:,actionIndex)), ...
        mean(unitTable.SelectivePolicyRelativeMAE),mean(difference),ciLower,ciUpper, ...
        sum(difference>0),sum(difference<0),sum(difference==0),W,rawP(actionIndex),nan};
end
holmP=holmAdjust(rawP);
for actionIndex=1:4, comparisonRows{actionIndex,13}=holmP(actionIndex); end
comparisonTable=cell2table(comparisonRows,'VariableNames', ...
    {'Comparator','IndependentUnitN','ComparatorRelativeMAE', ...
    'SelectivePolicyRelativeMAE','ComparatorMinusSelective','BootstrapCI_Lower', ...
    'BootstrapCI_Upper','UnitsFavourSelective','UnitsFavourComparator', ...
    'TiedUnits','SignedRankStatistic','RawP','HolmAdjustedP'});
comparisonTable.Comparator=string(comparisonTable.Comparator);

datasetRows=cell(3,13); datasets=["NUAA","PHM2010","Combined"];
for datasetIndex=1:3
    if datasets(datasetIndex)=="Combined"
        mask=true(height(unitTable),1);
    else
        mask=unitTable.Dataset==datasets(datasetIndex);
    end
    datasetRows(datasetIndex,:)={datasets(datasetIndex),sum(mask), ...
        mean(unitTable.FullRelativeMAE(mask)),mean(unitTable.MedianRelativeMAE(mask)), ...
        mean(unitTable.MaskDropoutRelativeMAE(mask)), ...
        mean(unitTable.ExclusionRelativeMAE(mask)), ...
        mean(unitTable.TrainingSelectedFixedRelativeMAE(mask)), ...
        mean(unitTable.LearnedPolicyRelativeMAE(mask)), ...
        mean(unitTable.SelectivePolicyRelativeMAE(mask)), ...
        mean(unitTable.OracleEnvelopeRelativeMAE(mask)), ...
        mean(unitTable.MeanRegretToOracle(mask)), ...
        mean(unitTable.SelectiveMeanRegretToOracle(mask)), ...
        mean(unitTable.CleanNonFullActionRate(mask))};
end
datasetTable=cell2table(datasetRows,'VariableNames', ...
    {'Dataset','IndependentUnitN','FullRelativeMAE','MedianRelativeMAE', ...
    'MaskDropoutRelativeMAE','ExclusionRelativeMAE', ...
    'TrainingSelectedFixedRelativeMAE','LearnedPolicyRelativeMAE', ...
    'SelectivePolicyRelativeMAE','OracleEnvelopeRelativeMAE', ...
    'MeanRegretToOracle','SelectiveMeanRegretToOracle','CleanNonFullActionRate'});
datasetTable.Dataset=string(datasetTable.Dataset);

configurationTable=table(trainRepeatCount,testRepeatCount,treeMaxSplits, ...
    detectorFloor,detectorQuantile,bootstrapCount,baseSeed,testSeedBase, ...
    runtimeSeconds,'VariableNames',{'TrainingTechnicalRepeats', ...
    'TestTechnicalRepeats','TreeMaxSplits','DetectorFloor', ...
    'DetectorQuantile','BootstrapResamples','TrainingSeedBase', ...
    'TestSeedBase','RuntimeSeconds'});

if runTransferAudits
    transferTable=cell2table(transferRows,'VariableNames', ...
        {'Dataset','Unit','HeldOutFaultFamily','ScenarioRecordN', ...
        'TrainingSelectedFixedAction','TrainingSelectedFixedRelativeMAE', ...
        'TransferredLearnedRelativeMAE','OracleRelativeMAE','ActionOverrideRate'});
    transferTable.Dataset=string(transferTable.Dataset);
    transferTable.Unit=string(transferTable.Unit);
    transferTable.HeldOutFaultFamily=string(transferTable.HeldOutFaultFamily);
    transferTable.TrainingSelectedFixedAction=string(transferTable.TrainingSelectedFixedAction);
    crossDatasetTable=cell2table(crossDatasetRows,'VariableNames', ...
        {'Dataset','Unit','TrainingDataset','ScenarioRecordN', ...
        'TrainingSelectedFixedAction','TrainingSelectedFixedRelativeMAE', ...
        'TransferredLearnedRelativeMAE','OracleRelativeMAE','ActionOverrideRate'});
    crossDatasetTable.Dataset=string(crossDatasetTable.Dataset);
    crossDatasetTable.Unit=string(crossDatasetTable.Unit);
    crossDatasetTable.TrainingDataset=string(crossDatasetTable.TrainingDataset);
    crossDatasetTable.TrainingSelectedFixedAction=string( ...
        crossDatasetTable.TrainingSelectedFixedAction);
    [transferUnitTable,transferPrimaryTable]=summariseTransferByUnit( ...
        transferTable,bootstrapCount,2026081701,"LeaveOneFaultFamilyOut");
    [crossUnitTable,crossPrimaryTable]=summariseTransferByUnit( ...
        crossDatasetTable,bootstrapCount,2026081702,"CrossDataset");
    transferRawP=[transferPrimaryTable.ExactP;crossPrimaryTable.ExactP];
    transferAdjustedP=holmAdjust(transferRawP);
    transferPrimaryTable.HolmAdjustedP=transferAdjustedP(1);
    crossPrimaryTable.HolmAdjustedP=transferAdjustedP(2);
    writetable(transferTable,fullfile(resultDir, ...
        'action_selector_leave_one_fault_family_rows.csv'));
    writetable(transferUnitTable,fullfile(resultDir, ...
        'action_selector_leave_one_fault_family_unit_results.csv'));
    writetable(transferPrimaryTable,fullfile(resultDir, ...
        'action_selector_leave_one_fault_family_primary.csv'));
    writetable(crossDatasetTable,fullfile(resultDir, ...
        'action_selector_cross_dataset_rows.csv'));
    writetable(crossUnitTable,fullfile(resultDir, ...
        'action_selector_cross_dataset_unit_results.csv'));
    writetable(crossPrimaryTable,fullfile(resultDir, ...
        'action_selector_cross_dataset_primary.csv'));
end

writetable(unitTable,fullfile(resultDir,'action_selector_unit_results.csv'));
writetable(primaryTable,fullfile(resultDir,'action_selector_primary_comparison.csv'));
writetable(comparisonTable,fullfile(resultDir,'action_selector_comparator_family.csv'));
writetable(datasetTable,fullfile(resultDir,'action_selector_dataset_summary.csv'));
writetable(predictionTable,fullfile(resultDir,'action_selector_outer_predictions.csv'));
writetable(configurationTable,fullfile(resultDir,'action_selector_configuration.csv'));
save(fullfile(resultDir,'crossfitted_action_selector_results.mat'), ...
    'unitTable','primaryTable','comparisonTable','datasetTable', ...
    'configurationTable','predictionTable','faultTypes','actionNames','-v7.3');
if runTransferAudits
    save(fullfile(resultDir,'action_selector_transfer_audits.mat'), ...
        'transferTable','transferUnitTable','transferPrimaryTable', ...
        'crossDatasetTable','crossUnitTable','crossPrimaryTable','-v7.3');
end

fprintf('\nPrimary learned-policy comparison:\n'); disp(primaryTable);
fprintf('\nComparator family:\n'); disp(comparisonTable);
fprintf('\nDataset summary:\n'); disp(datasetTable);
fprintf('Saved to %s\n',resultDir);

function P=prepareActionData(D,filePath,datasetName,rawSensorNames, ...
        displayNames,lambdaGrid,dropoutRates,baseSeed,detectorFloor, ...
        detectorQuantile)
    T=readtable(filePath,'VariableNamingRule','preserve');
    groupId=makeRawGroupId(T,datasetName);
    n=numel(D.y); nSensors=numel(rawSensorNames);
    rawSignals=cell(n,nSensors); cleanQuality=nan(n,nSensors,4);
    featureBlocks=cell(1,nSensors);
    for sensorIndex=1:nSensors
        position=find(D.sensorNames==rawSensorNames(sensorIndex),1);
        assert(~isempty(position),'Missing monitored sensor %s.',rawSensorNames(sensorIndex));
        featureBlocks{sensorIndex}=(position-1)*9+(1:9);
    end
    for recordIndex=1:n
        rows=groupId==D.runTable.SourceGroup(recordIndex);
        assert(any(rows),'No high-resolution samples for %s record %d.', ...
            datasetName,recordIndex);
        for sensorIndex=1:nSensors
            x=double(T.(char(rawSensorNames(sensorIndex)))(rows));
            x=x(isfinite(x)); rawSignals{recordIndex,sensorIndex}=x;
            cleanQuality(recordIndex,sensorIndex,:)=rawQualityDescriptors(x);
        end
    end
    outerPolicies=fitOuterPolicies(D,cleanQuality,featureBlocks,lambdaGrid, ...
        dropoutRates,baseSeed,detectorFloor,detectorQuantile);
    P.D=D; P.name=datasetName; P.rawSensorNames=rawSensorNames;
    P.displayNames=displayNames; P.rawSignals=rawSignals;
    P.featureBlocks=featureBlocks; P.cleanQuality=cleanQuality;
    P.outerPolicies=outerPolicies;
end

function policies=fitOuterPolicies(D,cleanQuality,featureBlocks,lambdaGrid, ...
        dropoutRates,baseSeed,detectorFloor,detectorQuantile)
    units=unique(D.experiment,'stable'); nSensors=numel(featureBlocks);
    Xfull=[D.processX,D.sensorX,D.timeX]; policies=cell(numel(units),1);
    for unitIndex=1:numel(units)
        isTrain=D.experiment~=units(unitIndex);
        fullLambda=groupedTuneLambda(Xfull(isTrain,:),D.y(isTrain), ...
            D.experiment(isTrain),lambdaGrid);
        sensorMedians=median(D.sensorX(isTrain,:),1,'omitnan');
        seed=baseSeed+unitIndex;
        [dropoutRate,dropoutLambda]=groupedTuneMaskDropout( ...
            Xfull(isTrain,:),D.y(isTrain),D.experiment(isTrain),featureBlocks, ...
            size(D.processX,2),lambdaGrid,dropoutRates,seed);
        dropLambdas=nan(1,nSensors);
        for sensorIndex=1:nSensors
            keep=true(1,size(D.sensorX,2)); keep(featureBlocks{sensorIndex})=false;
            Xdrop=[D.processX,D.sensorX(:,keep),D.timeX];
            dropLambdas(sensorIndex)=groupedTuneLambda(Xdrop(isTrain,:), ...
                D.y(isTrain),D.experiment(isTrain),lambdaGrid);
        end
        hyper=struct('fullLambda',fullLambda,'dropoutRate',dropoutRate, ...
            'dropoutLambda',dropoutLambda,'dropLambdas',dropLambdas);
        policy=fitFixedPolicy(D,cleanQuality,featureBlocks,isTrain,hyper,seed, ...
            detectorFloor,detectorQuantile);
        policy.unit=units(unitIndex); policies{unitIndex}=policy;
    end
end

function policy=fitFixedPolicy(D,cleanQuality,featureBlocks,isTrain,hyper, ...
        seed,detectorFloor,detectorQuantile)
    nSensors=numel(featureBlocks); Xfull=[D.processX,D.sensorX,D.timeX];
    policy.fullLambda=hyper.fullLambda; policy.dropoutRate=hyper.dropoutRate;
    policy.dropoutLambda=hyper.dropoutLambda; policy.dropLambdas=hyper.dropLambdas;
    policy.sensorMedians=median(D.sensorX(isTrain,:),1,'omitnan');
    policy.fullModel=fitRidge(Xfull(isTrain,:),D.y(isTrain),hyper.fullLambda);
    [Xaug,yaug]=augmentMaskDropoutTraining(Xfull(isTrain,:),D.y(isTrain), ...
        featureBlocks,size(D.processX,2),policy.sensorMedians, ...
        hyper.dropoutRate,seed);
    policy.dropoutModel=fitRidge(Xaug,yaug,hyper.dropoutLambda);
    policy.keepMasks=cell(1,nSensors); policy.dropModels=cell(1,nSensors);
    for sensorIndex=1:nSensors
        keep=true(1,size(D.sensorX,2)); keep(featureBlocks{sensorIndex})=false;
        Xdrop=[D.processX,D.sensorX(:,keep),D.timeX];
        policy.keepMasks{sensorIndex}=keep;
        policy.dropModels{sensorIndex}=fitRidge(Xdrop(isTrain,:),D.y(isTrain), ...
            hyper.dropLambdas(sensorIndex));
    end
    policy.detector=fitQualityDetector(D.sensorX(isTrain,:), ...
        cleanQuality(isTrain,:,:),featureBlocks,detectorFloor,detectorQuantile);
end

function rows=makeActionRows(P,unit,policy,faultTypes,repeatCount,seedBase,includeClean)
    D=P.D; unitMask=D.experiment==unit; globalIndex=find(unitMask);
    n=sum(unitMask); nFaults=numel(faultTypes); nSensors=numel(P.featureBlocks);
    totalRows=n*repeatCount*nFaults*nSensors+n*double(includeClean);
    numeric=zeros(totalRows,19); metadata=cell(totalRows,5); row=0;
    if includeClean
        [features,losses,flags]=actionFeaturesAndLosses(P,unitMask,policy, ...
            D.sensorX(unitMask,:),P.cleanQuality(unitMask,:,:));
        cleanScale=mean(losses(:,1));
        cleanScale=max(cleanScale,1e-12);
        losses=losses/cleanScale;
        for recordIndex=1:n
            row=row+1; numeric(row,:)=[features(recordIndex,:),losses(recordIndex,:), ...
                flags(recordIndex),D.y(globalIndex(recordIndex)),cleanScale];
            metadata(row,:)={P.name,unit,"Clean","None",0};
        end
    else
        [~,cleanLoss]=actionFeaturesAndLosses(P,unitMask,policy, ...
            D.sensorX(unitMask,:),P.cleanQuality(unitMask,:,:));
        cleanScale=max(mean(cleanLoss(:,1)),1e-12);
    end
    for repeatIndex=1:repeatCount
        for sensorIndex=1:nSensors
            for faultIndex=1:nFaults
                deterministicSeed=seedBase+datasetOffset(P.name)+ ...
                    unitNumericId(unit)*100000+repeatIndex*1000+ ...
                    sensorIndex*100+faultIndex;
                rng(deterministicSeed,'twister');
                Xvariant=D.sensorX(unitMask,:); quality=P.cleanQuality(unitMask,:,:);
                for localIndex=1:n
                    globalRecord=globalIndex(localIndex);
                    corrupted=injectRandomFault( ...
                        P.rawSignals{globalRecord,sensorIndex},faultTypes(faultIndex));
                    Xvariant(localIndex,P.featureBlocks{sensorIndex})= ...
                        signalFeatures(corrupted);
                    quality(localIndex,sensorIndex,:)=rawQualityDescriptors(corrupted);
                end
                [features,losses,flags]=actionFeaturesAndLosses(P,unitMask, ...
                    policy,Xvariant,quality);
                losses=losses/cleanScale;
                for localIndex=1:n
                    row=row+1; numeric(row,:)=[features(localIndex,:), ...
                        losses(localIndex,:),flags(localIndex), ...
                        D.y(globalIndex(localIndex)),cleanScale];
                    metadata(row,:)={P.name,unit,faultTypes(faultIndex), ...
                        P.displayNames(sensorIndex),repeatIndex};
                end
            end
        end
    end
    numeric=numeric(1:row,:); metadata=metadata(1:row,:);
    rows=array2table(numeric,'VariableNames', ...
        {'ScoreFz','ScoreVibrationX','ScoreVibrationY','MaximumScore', ...
        'SecondScore','ScoreGap','TriggerCount','PredictionRange', ...
        'PredictionSD','FullPrediction','CandidatePredictionMean', ...
        'DatasetPHM','FullLoss','MedianLoss','MaskDropoutLoss', ...
        'ExclusionLoss','DetectedSensorIndex','ActualWear','CleanFullMAE'});
    rows.Dataset=string(metadata(:,1)); rows.Unit=string(metadata(:,2));
    rows.FaultType=string(metadata(:,3)); rows.TargetSensor=string(metadata(:,4));
    rows.Repeat=cell2mat(metadata(:,5));
    rows.ConditionFamily=rows.TargetSensor+"|"+rows.FaultType;
end

function [features,losses,flags]=actionFeaturesAndLosses(P,unitMask,policy, ...
        Xvariant,quality)
    D=P.D; n=sum(unitMask); nSensors=numel(P.featureBlocks);
    Xfull=[D.processX(unitMask,:),Xvariant,D.timeX(unitMask,:)];
    fullPrediction=predictRidge(policy.fullModel,Xfull);
    scores=qualityNormalizedScores(Xvariant,quality,policy.detector,P.featureBlocks);
    [maximumScore,flags]=max(scores,[],2); flags(maximumScore<=1)=0;
    sortedScores=sort(scores,2,'descend'); secondScore=sortedScores(:,2);
    triggerCount=sum(scores>1,2);
    [Xmedian,maskMatrix]=applyDetectorReplacement(Xvariant,flags, ...
        P.featureBlocks,policy.sensorMedians);
    medianPrediction=predictRidge(policy.fullModel, ...
        [D.processX(unitMask,:),Xmedian,D.timeX(unitMask,:)]);
    maskPrediction=predictRidge(policy.dropoutModel, ...
        [D.processX(unitMask,:),Xmedian,D.timeX(unitMask,:),maskMatrix]);
    dropPrediction=nan(n,nSensors);
    for sensorIndex=1:nSensors
        keep=policy.keepMasks{sensorIndex};
        Xdrop=[D.processX(unitMask,:),Xvariant(:,keep),D.timeX(unitMask,:)];
        dropPrediction(:,sensorIndex)=predictRidge(policy.dropModels{sensorIndex},Xdrop);
    end
    exclusionPrediction=routePredictions(fullPrediction,dropPrediction,flags);
    candidate=[fullPrediction,medianPrediction,maskPrediction,exclusionPrediction];
    y=D.y(unitMask); losses=abs(candidate-y);
    features=[scores,maximumScore,secondScore,maximumScore-secondScore, ...
        triggerCount,range(candidate,2),std(candidate,0,2),fullPrediction, ...
        mean(candidate,2),repmat(double(P.name=="PHM2010"),n,1)];
end

function weights=balancedUnitFamilyWeights(T)
    group=findgroups(T.Unit,T.ConditionFamily); counts=splitapply(@numel,group,group);
    unitFamily=unique(T(:,{'Unit','ConditionFamily'}),'rows');
    [unitFamilyGroup,unitList]=findgroups(unitFamily.Unit);
    familyCounts=splitapply(@numel,unitFamilyGroup,unitFamilyGroup);
    weights=zeros(height(T),1);
    for row=1:height(T)
        unitIndex=find(unitList==T.Unit(row),1);
        familyCount=familyCounts(unitIndex);
        weights(row)=1/(numel(unitList)*familyCount*counts(group(row)));
    end
    weights=weights/sum(weights);
end

function models=fitActionModels(T,maxSplits)
    X=featureMatrix(T); Y=lossMatrix(T);
    Y=Y-min(Y,[],2); % learn action regret, not overall record difficulty
    minLeaf=max(20,round(0.002*height(T)));
    weights=T.RowWeight/sum(T.RowWeight);
    models=cell(1,4);
    for actionIndex=1:4
        models{actionIndex}=fitrtree(X,Y(:,actionIndex),'Weights',weights, ...
            'MaxNumSplits',maxSplits,'MinLeafSize',minLeaf, ...
            'PredictorSelection','allsplits','Surrogate','off');
    end
end

function index=selectBestFixedAction(T)
    Y=lossMatrix(T); degraded=T.FaultType~="Clean"; w=T.RowWeight(degraded);
    w=w/sum(w); meanLoss=sum(Y(degraded,:).*w,1); [~,index]=min(meanLoss);
end

function [index,predicted]=selectActions(models,T)
    X=featureMatrix(T); predicted=nan(height(T),4);
    for actionIndex=1:4, predicted(:,actionIndex)=predict(models{actionIndex},X); end
    [~,index]=min(predicted,[],2); index(T.TriggerCount==0)=1;
end

function threshold=selectSelectiveThreshold(T,fallbackIndex,maxSplits)
    units=unique(T.Unit,'stable'); predicted=nan(height(T),4);
    for unitIndex=1:numel(units)
        validation=T.Unit==units(unitIndex); training=~validation;
        models=fitActionModels(T(training,:),maxSplits);
        [~,foldPredicted]=selectActions(models,T(validation,:));
        predicted(validation,:)=foldPredicted;
    end
    [bestPredicted,bestIndex]=min(predicted,[],2);
    predictedGain=predicted(:,fallbackIndex)-bestPredicted;
    degraded=T.FaultType~="Clean";
    finiteGain=predictedGain(degraded & isfinite(predictedGain));
    if isempty(finiteGain), threshold=inf; return; end
    candidates=unique([0;prctile(finiteGain,[25 50 75 90 95])';inf]);
    actual=lossMatrix(T); unitLoss=nan(numel(units),numel(candidates));
    for candidateIndex=1:numel(candidates)
        selected=repmat(fallbackIndex,height(T),1);
        override=predictedGain>candidates(candidateIndex) & T.TriggerCount>0;
        selected(override)=bestIndex(override);
        chosen=actual(sub2ind(size(actual),(1:height(T))',selected));
        for unitIndex=1:numel(units)
            mask=degraded & T.Unit==units(unitIndex);
            weights=T.RowWeight(mask); weights=weights/sum(weights);
            unitLoss(unitIndex,candidateIndex)=sum(chosen(mask).*weights);
        end
    end
    meanLoss=mean(unitLoss,1);
    tolerance=1e-12*max(1,abs(min(meanLoss)));
    admissible=find(meanLoss<=min(meanLoss)+tolerance);
    threshold=candidates(admissible(end)); % conservative tie-break
end

function [index,gain]=selectSelectiveActions(predicted,T,fallbackIndex,threshold)
    [bestPredicted,bestIndex]=min(predicted,[],2);
    gain=predicted(:,fallbackIndex)-bestPredicted;
    index=repmat(fallbackIndex,height(T),1);
    override=gain>threshold & T.TriggerCount>0;
    index(override)=bestIndex(override);
end

function [unitTable,primaryTable]=summariseTransferByUnit(rowTable,B,seed,label)
    [group,dataset,unit]=findgroups(rowTable.Dataset,rowTable.Unit);
    fixed=splitapply(@mean,rowTable.TrainingSelectedFixedRelativeMAE,group);
    learned=splitapply(@mean,rowTable.TransferredLearnedRelativeMAE,group);
    oracle=splitapply(@mean,rowTable.OracleRelativeMAE,group);
    override=splitapply(@mean,rowTable.ActionOverrideRate,group);
    unitTable=table(dataset,unit,fixed,learned,oracle,override, ...
        'VariableNames',{'Dataset','Unit','TrainingSelectedFixedRelativeMAE', ...
        'TransferredLearnedRelativeMAE','OracleRelativeMAE','ActionOverrideRate'});
    difference=fixed-learned;
    [lower,upper]=clusterBootstrapMeanCI(difference,B,seed);
    [p,W]=exactSignedRank(difference);
    primaryTable=table(string(label),height(unitTable),mean(fixed),mean(learned), ...
        mean(difference),lower,upper,sum(difference>0),sum(difference<0), ...
        sum(difference==0),W,p,'VariableNames',{'Audit','IndependentUnitN', ...
        'TrainingSelectedFixedRelativeMAE','TransferredLearnedRelativeMAE', ...
        'FixedMinusTransferred','BootstrapCI_Lower','BootstrapCI_Upper', ...
        'UnitsFavourTransferred','UnitsFavourFixed','TiedUnits', ...
        'SignedRankStatistic','ExactP'});
end

function X=featureMatrix(T)
    X=T{:,{'ScoreFz','ScoreVibrationX','ScoreVibrationY','MaximumScore', ...
        'SecondScore','ScoreGap','TriggerCount','PredictionRange', ...
        'PredictionSD','FullPrediction','CandidatePredictionMean','DatasetPHM'}};
end

function Y=lossMatrix(T)
    Y=T{:,{'FullLoss','MedianLoss','MaskDropoutLoss','ExclusionLoss'}};
end

function value=datasetOffset(name)
    if name=="NUAA", value=10000000; else, value=20000000; end
end

function value=unitNumericId(unit)
    digits=regexp(char(unit),'\d+','match','once'); value=str2double(digits);
    if ~isfinite(value), value=sum(double(char(unit))); end
end

function [p,W]=exactSignedRank(difference)
    difference=difference(isfinite(difference) & difference~=0);
    if isempty(difference), p=1; W=0; return; end
    [p,~,stats]=signrank(difference,0,'method','exact'); W=stats.signedrank;
end

function [lower,upper]=clusterBootstrapMeanCI(difference,B,seed)
    difference=difference(isfinite(difference)); n=numel(difference);
    stream=RandStream('mt19937ar','Seed',seed); boot=nan(B,1);
    for b=1:B, boot(b)=mean(difference(randi(stream,n,n,1))); end
    q=prctile(boot,[2.5 97.5]); lower=q(1); upper=q(2);
end

function adjusted=holmAdjust(p)
    [sorted,index]=sort(p(:)); m=numel(p); sortedAdjusted=zeros(m,1);
    for i=1:m, sortedAdjusted(i)=min(1,(m-i+1)*sorted(i)); end
    sortedAdjusted=cummax(sortedAdjusted); adjusted=zeros(m,1);
    adjusted(index)=sortedAdjusted;
end

function groupId=makeRawGroupId(T,datasetName)
    tag=string(T.experiment_tag); timestamp=double(T.timestamp);
    if datasetName=="NUAA"
        [groupId,~,~]=findgroups(tag,double(T.experiment_csv_n));
    else
        groupId=zeros(height(T),1); nextGroup=0; tags=unique(tag,'stable');
        for tagIndex=1:numel(tags)
            rows=find(tag==tags(tagIndex)); [sortedTimestamp,order]=sort(timestamp(rows));
            rows=rows(order); increments=diff(sortedTimestamp);
            cadence=median(increments(increments>0),'omitnan');
            localRun=cumsum([true;increments<0.1*cadence]);
            groupId(rows)=nextGroup+localRun; nextGroup=nextGroup+max(localRun);
        end
    end
end

function q=rawQualityDescriptors(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), q=[1 1 1 0]; return; end
    tolerance=max(1e-12,1e-8*max(1,max(abs(x))));
    rail=max(mean(abs(x-min(x))<=tolerance),mean(abs(x-max(x))<=tolerance));
    zero=mean(abs(x)<=tolerance);
    if numel(x)>1
        dx=diff(x); flat=mean(abs(dx)<=tolerance); diffRms=sqrt(mean(dx.^2));
    else
        flat=1; diffRms=0;
    end
    q=[rail,zero,flat,diffRms/max(std(x,0),1e-10)]; q(~isfinite(q))=0;
end

function f=signalFeatures(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), f=zeros(1,9); return; end
    mu=mean(x); rmsValue=sqrt(mean(x.^2)); sigma=std(x,0);
    peakToPeak=max(x)-min(x); med=median(x); madValue=median(abs(x-med));
    if sigma>max(eps(abs(mu)),1e-12)
        z=(x-mu)./sigma; skewValue=mean(z.^3); kurtValue=mean(z.^4);
    else
        skewValue=0; kurtValue=0;
    end
    crestFactor=max(abs(x))/max(rmsValue,eps);
    f=[mu,rmsValue,sigma,peakToPeak,skewValue,kurtValue,med,madValue,crestFactor];
    f(~isfinite(f))=0;
end

function xc=injectRandomFault(x,faultType)
    xc=double(x(:)); n=numel(xc); scale=std(xc,0,'omitnan');
    if ~isfinite(scale) || scale<1e-10
        scale=max(1e-6,0.01*max(abs(xc),[],'omitnan'));
    end
    if faultType=="Saturation" || faultType=="Dropout"
        duration=0.10+0.60*rand; severity=duration;
    elseif faultType=="SoftClipping"
        duration=0.20+0.60*rand; severity=0.50+2.50*rand;
    elseif faultType=="IntermittentDropout"
        duration=0.10+0.40*rand; severity=duration;
    elseif faultType=="GradualDrift"
        duration=0.40+0.60*rand; severity=0.50+2.50*rand;
    elseif faultType=="HeteroscedasticNoise"
        duration=0.40+0.60*rand; severity=0.50+1.50*rand;
    elseif faultType=="Bias"
        duration=0.20+0.80*rand; severity=0.50+2.50*rand;
    else
        duration=0.20+0.80*rand; severity=0.25+1.75*rand;
    end
    blockLength=min(n,max(1,round(duration*n)));
    startIndex=randi(max(1,n-blockLength+1)); idx=startIndex:(startIndex+blockLength-1);
    switch faultType
        case "Saturation"
            xc(idx)=max(xc,[],'omitnan');
        case "Dropout"
            xc(idx)=0;
        case "Bias"
            direction=2*(rand>0.5)-1; xc(idx)=xc(idx)+direction*severity*scale;
        case "Noise"
            xc(idx)=xc(idx)+severity*scale*randn(blockLength,1);
        case "SoftClipping"
            center=median(xc(idx),'omitnan'); limit=scale/max(severity,1e-6);
            xc(idx)=center+limit*tanh((xc(idx)-center)/limit);
        case "IntermittentDropout"
            burstN=randi([3 8]); target=max(burstN,round(duration*n));
            burstLength=max(1,round(target/burstN)); affected=false(n,1);
            for b=1:burstN
                start=randi(max(1,n-burstLength+1));
                affected(start:(start+burstLength-1))=true;
            end
            xc(affected)=0;
        case "GradualDrift"
            direction=2*(rand>0.5)-1; ramp=linspace(0,1,blockLength)';
            xc(idx)=xc(idx)+direction*severity*scale*ramp;
        case "HeteroscedasticNoise"
            localScale=linspace(0.05,severity,blockLength)';
            xc(idx)=xc(idx)+scale*localScale.*randn(blockLength,1);
    end
end

function detector=fitQualityDetector(Xsensor,cleanQuality,featureBlocks, ...
        detectorFloor,detectorQuantile)
    selectedStats=[1 2 3 4 7 8 9]; nSensors=numel(featureBlocks);
    detector.center=cell(1,nSensors); detector.scale=cell(1,nSensors);
    detector.threshold=zeros(1,nSensors);
    for sensorIndex=1:nSensors
        rawQ=squeeze(cleanQuality(:,sensorIndex,:));
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        center=median(Q,1,'omitnan'); scale=1.4826*median(abs(Q-center),1,'omitnan');
        fallback=std(Q,0,1,'omitnan'); bad=~isfinite(scale)|scale<1e-9;
        scale(bad)=fallback(bad); scale(~isfinite(scale)|scale<1e-9)=1e-9;
        score=max(abs((Q-center)./scale),[],2);
        detector.center{sensorIndex}=center; detector.scale{sensorIndex}=scale;
        detector.threshold(sensorIndex)=max(detectorFloor,prctile(score,detectorQuantile));
    end
end

function score=qualityNormalizedScores(Xsensor,quality,detector,featureBlocks)
    selectedStats=[1 2 3 4 7 8 9]; n=size(Xsensor,1);
    score=zeros(n,numel(featureBlocks));
    for sensorIndex=1:numel(featureBlocks)
        rawQ=squeeze(quality(:,sensorIndex,:));
        if n==1, rawQ=reshape(rawQ,1,[]); end
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        rawScore=max(abs((Q-detector.center{sensorIndex})./ ...
            detector.scale{sensorIndex}),[],2);
        score(:,sensorIndex)=rawScore./detector.threshold(sensorIndex);
    end
end

function [Xreplaced,maskMatrix]=applyDetectorReplacement(Xsensor,flags, ...
        featureBlocks,sensorMedians)
    Xreplaced=Xsensor; maskMatrix=zeros(size(Xsensor,1),numel(featureBlocks));
    for sensorIndex=1:numel(featureBlocks)
        rows=flags==sensorIndex;
        if any(rows)
            Xreplaced(rows,featureBlocks{sensorIndex})=repmat( ...
                sensorMedians(featureBlocks{sensorIndex}),sum(rows),1);
            maskMatrix(rows,sensorIndex)=1;
        end
    end
end

function routed=routePredictions(fullPrediction,dropPrediction,flags)
    routed=fullPrediction;
    for sensorIndex=1:size(dropPrediction,2)
        rows=flags==sensorIndex; routed(rows)=dropPrediction(rows,sensorIndex);
    end
end

function [bestRate,bestLambda]=groupedTuneMaskDropout(X,y,groups, ...
        featureBlocks,nProcess,lambdaGrid,dropoutRates,baseSeed)
    units=unique(groups,'stable'); scores=nan(numel(dropoutRates),numel(lambdaGrid));
    for rateIndex=1:numel(dropoutRates)
        for lambdaIndex=1:numel(lambdaGrid)
            foldMAE=nan(numel(units),1);
            for unitIndex=1:numel(units)
                isValidation=groups==units(unitIndex); isTrain=~isValidation;
                medians=median(X(isTrain,:),1,'omitnan');
                [Xaug,yaug]=augmentMaskDropoutTraining(X(isTrain,:),y(isTrain), ...
                    featureBlocks,nProcess,medians(nProcess+(1:size(X,2)-nProcess-2)), ...
                    dropoutRates(rateIndex),baseSeed+1000*unitIndex+100*rateIndex);
                model=fitRidge(Xaug,yaug,lambdaGrid(lambdaIndex));
                Xvalidation=X(isValidation,:); errors=[];
                cleanMask=zeros(sum(isValidation),numel(featureBlocks));
                errors=[errors;abs(predictRidge(model,[Xvalidation,cleanMask])-y(isValidation))]; %#ok<AGROW>
                for sensorIndex=1:numel(featureBlocks)
                    Xstate=Xvalidation;
                    columns=nProcess+featureBlocks{sensorIndex};
                    Xstate(:,columns)=repmat(medians(columns),sum(isValidation),1);
                    maskState=zeros(sum(isValidation),numel(featureBlocks));
                    maskState(:,sensorIndex)=1;
                    errors=[errors;abs(predictRidge(model,[Xstate,maskState])-y(isValidation))]; %#ok<AGROW>
                end
                foldMAE(unitIndex)=mean(errors);
            end
            scores(rateIndex,lambdaIndex)=mean(foldMAE);
        end
    end
    [~,linear]=min(scores,[],'all','linear');
    [rateIndex,lambdaIndex]=ind2sub(size(scores),linear);
    bestRate=dropoutRates(rateIndex); bestLambda=lambdaGrid(lambdaIndex);
end

function [Xaug,yaug]=augmentMaskDropoutTraining(X,y,featureBlocks,nProcess, ...
        sensorMedians,dropoutRate,seed)
    n=size(X,1); nSensors=numel(featureBlocks); stream=RandStream('mt19937ar','Seed',seed);
    selected=find(rand(stream,n,1)<dropoutRate);
    if isempty(selected), selected=randi(stream,n,1,1); end
    selectedSensor=randi(stream,nSensors,numel(selected),1);
    clean=[X,zeros(n,nSensors)]; masked=X(selected,:); masks=zeros(numel(selected),nSensors);
    for row=1:numel(selected)
        sensorIndex=selectedSensor(row); columns=nProcess+featureBlocks{sensorIndex};
        masked(row,columns)=sensorMedians(featureBlocks{sensorIndex});
        masks(row,sensorIndex)=1;
    end
    Xaug=[clean;masked,masks]; yaug=[y(:);y(selected)];
end

function lambda=groupedTuneLambda(X,y,groups,lambdaGrid)
    units=unique(groups,'stable'); scores=nan(numel(lambdaGrid),1);
    for lambdaIndex=1:numel(lambdaGrid)
        foldMAE=nan(numel(units),1);
        for unitIndex=1:numel(units)
            isValidation=groups==units(unitIndex); isTrain=~isValidation;
            model=fitRidge(X(isTrain,:),y(isTrain),lambdaGrid(lambdaIndex));
            foldMAE(unitIndex)=mean(abs(predictRidge(model,X(isValidation,:))-y(isValidation)));
        end
        scores(lambdaIndex)=mean(foldMAE);
    end
    [~,best]=min(scores); lambda=lambdaGrid(best);
end

function model=fitRidge(X,y,lambda)
    X=double(X); y=double(y(:)); muX=mean(X,1,'omitnan'); muX(~isfinite(muX))=0;
    sigmaX=std(X,0,1,'omitnan'); sigmaX(~isfinite(sigmaX)|sigmaX<1e-12)=1;
    Z=(X-muX)./sigmaX; keep=std(Z,0,1)>1e-12; Z=Z(:,keep);
    muY=mean(y); yc=y-muY;
    beta=(Z'*Z+lambda*eye(size(Z,2)))\(Z'*yc);
    model.muX=muX; model.sigmaX=sigmaX; model.keep=keep;
    model.beta=beta; model.muY=muY;
end

function yhat=predictRidge(model,X)
    Z=(double(X)-model.muX)./model.sigmaX;
    yhat=model.muY+Z(:,model.keep)*model.beta;
end
