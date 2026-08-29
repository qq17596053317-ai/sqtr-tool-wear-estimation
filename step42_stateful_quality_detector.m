clear; clc; close all;

paperRoot=fileparts(mfilename('fullpath'));
dataRoot=string(getenv('SQTR_RAW_DATA_DIR'));
if strlength(dataRoot)==0, dataRoot="C:\Users\66485\Desktop\论文"; end
resultDir=string(getenv('SQTR_STATEFUL_RESULT_DIR'));
if strlength(resultDir)==0
    resultDir=fullfile(paperRoot,'targeted_reviewer_experiments', ...
        'stateful_quality_detector','results');
end
if ~exist(resultDir,'dir'), mkdir(resultDir); end
trainRepeatCount=str2double(getenv('SQTR_STATEFUL_TRAIN_REPEATS'));
if ~isfinite(trainRepeatCount), trainRepeatCount=3; end
testRepeatCount=str2double(getenv('SQTR_STATEFUL_TEST_REPEATS'));
if ~isfinite(testRepeatCount), testRepeatCount=30; end

faultModes=["LinearBiasDrift","GainDrift","ColoredNoiseRamp", ...
    "IntermittentRecordDropout"];
rhoGrid=[0.50 0.70 0.85]; ewmaThresholdGrid=[0.8 1.0 1.2];
cusumReferenceGrid=[0.4 0.6 0.8]; cusumThresholdGrid=[1 2 4 6];
consecutiveGrid=[1 2 3]; releaseFraction=0.8;
detectorFloor=4; detectorQuantile=99; bootstrapCount=10000;
trainSeedBase=2026087000; testSeedBase=2026090000;

S=load(fullfile(paperRoot,'external_validation_results', ...
    'external_validation_complete_results.mat'),'nuaa','phm','lambdaGrid');
fprintf('Preparing grouped full and exclusion branches...\n');
nuaa=prepareStatefulData(S.nuaa, ...
    fullfile(dataRoot,'nuaa_orthogonal_bundle_high_resolution.csv'), ...
    "NUAA",["force_z","vibration1","vibration2"], ...
    ["force_z","vibration_x","vibration_y"],S.lambdaGrid, ...
    detectorFloor,detectorQuantile);
phm=prepareStatefulData(S.phm, ...
    fullfile(dataRoot,'phm2010_bundle_high_resolution.csv'), ...
    "PHM2010",["force_z","vibration_x","vibration_y"], ...
    ["force_z","vibration_x","vibration_y"],S.lambdaGrid, ...
    detectorFloor,detectorQuantile);
prepared={nuaa,phm};

scenarioTables=cell(12,1); unitRows=cell(12,21); configRows=cell(12,10);
unitRow=0; totalTimer=tic;
for datasetIndex=1:2
    P=prepared{datasetIndex}; units=unique(P.D.experiment,'stable');
    for unitIndex=1:numel(units)
        testUnit=units(unitIndex); policy=P.outerPolicies{unitIndex};
        fprintf('Stateful detector outer unit %s | %s (%d/%d)\n', ...
            P.name,testUnit,unitIndex,numel(units));
        tuningSequences=collectTuningSequences(P,testUnit,policy,faultModes, ...
            trainRepeatCount,trainSeedBase);
        selectedConfig=selectStatefulConfig(tuningSequences,rhoGrid, ...
            ewmaThresholdGrid,cusumReferenceGrid,cusumThresholdGrid, ...
            consecutiveGrid,releaseFraction);
        testScenarios=runStatefulTestUnit(P,testUnit,policy,faultModes, ...
            testRepeatCount,testSeedBase,selectedConfig,releaseFraction);
        scenarioTables{unitRow+1}=testScenarios;
        degraded=testScenarios.FaultMode~="Clean";
        pointMAE=mean(testScenarios.PointRelativeMAE(degraded));
        statefulMAE=mean(testScenarios.StatefulRelativeMAE(degraded));
        fullMAE=mean(testScenarios.FullRelativeMAE(degraded));
        oracleMAE=mean(testScenarios.OracleRelativeMAE(degraded));
        pointEvent=mean(testScenarios.PointEventDetected(degraded));
        statefulEvent=mean(testScenarios.StatefulEventDetected(degraded));
        pointDelay=mean(testScenarios.PointDetectionDelay(degraded),'omitnan');
        statefulDelay=mean(testScenarios.StatefulDetectionDelay(degraded),'omitnan');
        clean=~degraded;
        unitRow=unitRow+1;
        unitRows(unitRow,:)={P.name,testUnit,testRepeatCount, ...
            selectedConfig.Method,selectedConfig.Parameter,selectedConfig.Threshold, ...
            selectedConfig.Consecutive,fullMAE,pointMAE,statefulMAE,oracleMAE, ...
            pointMAE-statefulMAE,pointEvent,statefulEvent,pointDelay,statefulDelay, ...
            mean(testScenarios.PointPreOnsetFalseRate(degraded)), ...
            mean(testScenarios.StatefulPreOnsetFalseRate(degraded)), ...
            mean(testScenarios.PointTriggerRate(clean)), ...
            mean(testScenarios.StatefulTriggerRate(clean)), ...
            mean(testScenarios.StatefulSwitchCount(clean))};
        configRows(unitRow,:)={P.name,testUnit,selectedConfig.Method, ...
            selectedConfig.Parameter,selectedConfig.Threshold,selectedConfig.Consecutive, ...
            selectedConfig.TrainingCleanTriggerRate, ...
            selectedConfig.TrainingEventDetectionRate, ...
            selectedConfig.TrainingMeanDelay,selectedConfig.Feasible};
    end
end
runtimeSeconds=toc(totalTimer);

scenarioTable=vertcat(scenarioTables{:});
unitTable=cell2table(unitRows,'VariableNames', ...
    {'Dataset','Unit','TechnicalRepeatN','SelectedStateMethod', ...
    'SelectedStateParameter','SelectedThreshold', ...
    'SelectedConsecutiveRecords','FullRelativeMAE','PointRelativeMAE', ...
    'StatefulRelativeMAE','OracleRelativeMAE','PointMinusStateful', ...
    'PointEventDetectionRate','StatefulEventDetectionRate', ...
    'PointMeanDetectionDelay','StatefulMeanDetectionDelay', ...
    'PointPreOnsetFalseRate','StatefulPreOnsetFalseRate', ...
    'PointCleanTriggerRate','StatefulCleanTriggerRate','StatefulCleanSwitchCount'});
unitTable.Dataset=string(unitTable.Dataset); unitTable.Unit=string(unitTable.Unit);
unitTable.SelectedStateMethod=string(unitTable.SelectedStateMethod);
configTable=cell2table(configRows,'VariableNames', ...
    {'Dataset','Unit','Method','Parameter','Threshold','ConsecutiveRecords', ...
    'TrainingCleanTriggerRate','TrainingEventDetectionRate', ...
    'TrainingMeanDelay','MetFivePercentCleanConstraint'});
configTable.Dataset=string(configTable.Dataset); configTable.Unit=string(configTable.Unit);
configTable.Method=string(configTable.Method);

difference=unitTable.PointMinusStateful;
[ciLower,ciUpper]=clusterBootstrapMeanCI(difference,bootstrapCount,2026091591);
[p,W]=exactSignedRank(difference);
primaryTable=table(height(unitTable),trainRepeatCount,testRepeatCount, ...
    mean(unitTable.PointRelativeMAE),mean(unitTable.StatefulRelativeMAE), ...
    mean(difference),ciLower,ciUpper,sum(difference>0),sum(difference<0), ...
    sum(difference==0),W,p,'VariableNames', ...
    {'IndependentUnitN','TrainingTechnicalRepeats','TestTechnicalRepeats', ...
    'PointRelativeMAE','StatefulRelativeMAE','PointMinusStateful', ...
    'BootstrapCI_Lower','BootstrapCI_Upper','UnitsFavourStateful', ...
    'UnitsFavourPoint','TiedUnits','SignedRankStatistic','ExactP'});

datasets=["NUAA","PHM2010","Combined"]; datasetRows=cell(3,11);
for i=1:3
    if datasets(i)=="Combined", mask=true(height(unitTable),1);
    else, mask=unitTable.Dataset==datasets(i); end
    datasetRows(i,:)={datasets(i),sum(mask),mean(unitTable.FullRelativeMAE(mask)), ...
        mean(unitTable.PointRelativeMAE(mask)),mean(unitTable.StatefulRelativeMAE(mask)), ...
        mean(unitTable.OracleRelativeMAE(mask)), ...
        mean(unitTable.PointEventDetectionRate(mask)), ...
        mean(unitTable.StatefulEventDetectionRate(mask)), ...
        mean(unitTable.PointMeanDetectionDelay(mask)), ...
        mean(unitTable.StatefulMeanDetectionDelay(mask)), ...
        mean(unitTable.StatefulCleanTriggerRate(mask))};
end
datasetTable=cell2table(datasetRows,'VariableNames', ...
    {'Dataset','IndependentUnitN','FullRelativeMAE','PointRelativeMAE', ...
    'StatefulRelativeMAE','OracleRelativeMAE','PointEventDetectionRate', ...
    'StatefulEventDetectionRate','PointMeanDetectionDelay', ...
    'StatefulMeanDetectionDelay','StatefulCleanTriggerRate'});
datasetTable.Dataset=string(datasetTable.Dataset);

configurationTable=table(trainRepeatCount,testRepeatCount,releaseFraction, ...
    detectorFloor,detectorQuantile,bootstrapCount,trainSeedBase,testSeedBase, ...
    runtimeSeconds,'VariableNames',{'TrainingTechnicalRepeats', ...
    'TestTechnicalRepeats','ReleaseThresholdFraction','DetectorFloor', ...
    'DetectorQuantile','BootstrapResamples','TrainingSeedBase','TestSeedBase', ...
    'RuntimeSeconds'});

writetable(scenarioTable,fullfile(resultDir,'stateful_detector_scenarios.csv'));
writetable(unitTable,fullfile(resultDir,'stateful_detector_unit_results.csv'));
writetable(configTable,fullfile(resultDir,'stateful_detector_selected_configs.csv'));
writetable(primaryTable,fullfile(resultDir,'stateful_detector_primary_comparison.csv'));
writetable(datasetTable,fullfile(resultDir,'stateful_detector_dataset_summary.csv'));
writetable(configurationTable,fullfile(resultDir,'stateful_detector_configuration.csv'));
save(fullfile(resultDir,'stateful_detector_results.mat'),'scenarioTable', ...
    'unitTable','configTable','primaryTable','datasetTable','configurationTable', ...
    'faultModes','rhoGrid','ewmaThresholdGrid','cusumReferenceGrid', ...
    'cusumThresholdGrid','consecutiveGrid','-v7.3');

fprintf('\nStateful detector primary comparison:\n'); disp(primaryTable);
fprintf('\nDataset summary:\n'); disp(datasetTable);
fprintf('Saved to %s\n',resultDir);

function P=prepareStatefulData(D,filePath,datasetName,rawSensorNames, ...
        displayNames,lambdaGrid,detectorFloor,detectorQuantile)
    T=readtable(filePath,'VariableNamingRule','preserve');
    groupId=makeRawGroupId(T,datasetName); n=numel(D.y);
    rawSignals=cell(n,3); cleanQuality=nan(n,3,4); featureBlocks=cell(1,3);
    for sensorIndex=1:3
        position=find(D.sensorNames==rawSensorNames(sensorIndex),1);
        featureBlocks{sensorIndex}=(position-1)*9+(1:9);
    end
    for recordIndex=1:n
        rows=groupId==D.runTable.SourceGroup(recordIndex);
        for sensorIndex=1:3
            x=double(T.(char(rawSensorNames(sensorIndex)))(rows)); x=x(isfinite(x));
            rawSignals{recordIndex,sensorIndex}=x;
            cleanQuality(recordIndex,sensorIndex,:)=rawQualityDescriptors(x);
        end
    end
    units=unique(D.experiment,'stable'); policies=cell(numel(units),1);
    Xfull=[D.processX,D.sensorX,D.timeX];
    for unitIndex=1:numel(units)
        train=D.experiment~=units(unitIndex);
        fullLambda=groupedTuneLambda(Xfull(train,:),D.y(train), ...
            D.experiment(train),lambdaGrid);
        policy.fullModel=fitRidge(Xfull(train,:),D.y(train),fullLambda);
        policy.keepMasks=cell(1,3); policy.dropModels=cell(1,3);
        for sensorIndex=1:3
            keep=true(1,size(D.sensorX,2)); keep(featureBlocks{sensorIndex})=false;
            Xdrop=[D.processX,D.sensorX(:,keep),D.timeX];
            lambda=groupedTuneLambda(Xdrop(train,:),D.y(train), ...
                D.experiment(train),lambdaGrid);
            policy.keepMasks{sensorIndex}=keep;
            policy.dropModels{sensorIndex}=fitRidge(Xdrop(train,:),D.y(train),lambda);
        end
        policy.detector=fitQualityDetector(D.sensorX(train,:), ...
            cleanQuality(train,:,:),featureBlocks,detectorFloor,detectorQuantile);
        policies{unitIndex}=policy;
    end
    P.D=D; P.name=datasetName; P.rawSignals=rawSignals;
    P.cleanQuality=cleanQuality; P.featureBlocks=featureBlocks;
    P.displayNames=displayNames; P.outerPolicies=policies;
end

function sequences=collectTuningSequences(P,testUnit,policy,faultModes, ...
        repeatCount,seedBase)
    units=unique(P.D.experiment,'stable'); units=units(units~=testUnit);
    rows=cell(numel(units)*(1+repeatCount*3*numel(faultModes)),1); row=0;
    for unitIndex=1:numel(units)
        mask=P.D.experiment==units(unitIndex);
        cleanScores=qualityNormalizedScores(P.D.sensorX(mask,:), ...
            P.cleanQuality(mask,:,:),policy.detector,P.featureBlocks);
        row=row+1; rows{row}=struct('Scores',cleanScores,'Target',0, ...
            'Onset',inf,'Clean',true);
        for repeatIndex=1:repeatCount
            for sensorIndex=1:3
                for modeIndex=1:numel(faultModes)
                    seed=seedBase+datasetOffset(P.name)+unitNumericId(units(unitIndex))*100000+ ...
                        repeatIndex*1000+sensorIndex*100+modeIndex;
                    [Xvariant,quality,onset]=makeSequentialFault(P,mask,sensorIndex, ...
                        faultModes(modeIndex),seed);
                    scores=qualityNormalizedScores(Xvariant,quality,policy.detector, ...
                        P.featureBlocks);
                    row=row+1; rows{row}=struct('Scores',scores,'Target',sensorIndex, ...
                        'Onset',onset,'Clean',false);
                end
            end
        end
    end
    sequences=rows(1:row);
end

function selected=selectStatefulConfig(sequences,rhoGrid,ewmaThresholdGrid, ...
        cusumReferenceGrid,cusumThresholdGrid,consecutiveGrid,releaseFraction)
    methods=strings(0,1); parameter=[]; thresholdValue=[]; consecutiveValue=[];
    cleanValue=[]; eventValue=[]; delayValueList=[];
    for method=["EWMA","CUSUM"]
        if method=="EWMA"
            parameterGrid=rhoGrid; currentThresholdGrid=ewmaThresholdGrid;
        else
            parameterGrid=cusumReferenceGrid; currentThresholdGrid=cusumThresholdGrid;
        end
        for currentParameter=parameterGrid
            for threshold=currentThresholdGrid
                for consecutive=consecutiveGrid
                    cleanRates=[]; event=[]; delays=[];
                    for i=1:numel(sequences)
                        S=sequences{i}; flags=statefulFlags(S.Scores,method, ...
                            currentParameter,threshold,consecutive,releaseFraction);
                        if S.Clean
                            cleanRates(end+1,1)=mean(flags>0); %#ok<AGROW>
                        else
                            after=(S.Onset:size(S.Scores,1))';
                            hit=after(flags(after)==S.Target);
                            event(end+1,1)=~isempty(hit); %#ok<AGROW>
                            if isempty(hit)
                                delays(end+1,1)=size(S.Scores,1)-S.Onset+1; %#ok<AGROW>
                            else
                                delays(end+1,1)=hit(1)-S.Onset; %#ok<AGROW>
                            end
                        end
                    end
                    methods(end+1,1)=method; %#ok<AGROW>
                    parameter(end+1,1)=currentParameter; %#ok<AGROW>
                    thresholdValue(end+1,1)=threshold; %#ok<AGROW>
                    consecutiveValue(end+1,1)=consecutive; %#ok<AGROW>
                    cleanValue(end+1,1)=mean(cleanRates); %#ok<AGROW>
                    eventValue(end+1,1)=mean(event); %#ok<AGROW>
                    delayValueList(end+1,1)=mean(delays); %#ok<AGROW>
                end
            end
        end
    end
    feasible=cleanValue<=0.05;
    if any(feasible), pool=find(feasible); else, pool=(1:numel(cleanValue))'; end
    methodPreference=double(methods(pool)=="EWMA");
    ranking=[-eventValue(pool),delayValueList(pool),cleanValue(pool), ...
        -thresholdValue(pool),methodPreference,parameter(pool),consecutiveValue(pool)];
    [~,order]=sortrows(ranking,1:size(ranking,2)); best=pool(order(1));
    selected=struct('Method',methods(best),'Parameter',parameter(best), ...
        'Threshold',thresholdValue(best),'Consecutive',consecutiveValue(best), ...
        'TrainingCleanTriggerRate',cleanValue(best), ...
        'TrainingEventDetectionRate',eventValue(best), ...
        'TrainingMeanDelay',delayValueList(best),'Feasible',feasible(best));
end

function T=runStatefulTestUnit(P,unit,policy,faultModes,repeatCount, ...
        seedBase,config,releaseFraction)
    mask=P.D.experiment==unit; n=sum(mask); cleanScale=cleanFullMAE(P,mask,policy);
    rows=cell(1+repeatCount*3*numel(faultModes),18); row=0;
    cleanScores=qualityNormalizedScores(P.D.sensorX(mask,:),P.cleanQuality(mask,:,:), ...
        policy.detector,P.featureBlocks);
    pointFlags=pointFlagsFromScores(cleanScores);
    stateFlags=statefulFlags(cleanScores,config.Method,config.Parameter, ...
        config.Threshold,config.Consecutive,releaseFraction);
    row=row+1; rows(row,:)={P.name,unit,0,"None","Clean",n,inf, ...
        1,1,1,1,false,false,nan,nan,mean(pointFlags>0), ...
        mean(stateFlags>0),countSwitches(stateFlags)};
    for repeatIndex=1:repeatCount
        for sensorIndex=1:3
            for modeIndex=1:numel(faultModes)
                seed=seedBase+datasetOffset(P.name)+unitNumericId(unit)*100000+ ...
                    repeatIndex*1000+sensorIndex*100+modeIndex;
                [Xvariant,quality,onset]=makeSequentialFault(P,mask,sensorIndex, ...
                    faultModes(modeIndex),seed);
                scores=qualityNormalizedScores(Xvariant,quality,policy.detector, ...
                    P.featureBlocks);
                pointFlags=pointFlagsFromScores(scores);
                stateFlags=statefulFlags(scores,config.Method,config.Parameter, ...
                    config.Threshold,config.Consecutive,releaseFraction);
                [fullPred,dropPred]=branchPredictions(P,mask,policy,Xvariant);
                pointPred=routePredictions(fullPred,dropPred,pointFlags);
                statePred=routePredictions(fullPred,dropPred,stateFlags);
                oracleFlags=zeros(n,1); oracleFlags(onset:end)=sensorIndex;
                oraclePred=routePredictions(fullPred,dropPred,oracleFlags);
                after=onset:n; y=P.D.y(mask);
                pointHit=find(pointFlags(after)==sensorIndex,1);
                stateHit=find(stateFlags(after)==sensorIndex,1);
                row=row+1; rows(row,:)={P.name,unit,repeatIndex, ...
                    P.displayNames(sensorIndex),faultModes(modeIndex),n,onset, ...
                    mean(abs(fullPred(after)-y(after)))/cleanScale, ...
                    mean(abs(pointPred(after)-y(after)))/cleanScale, ...
                    mean(abs(statePred(after)-y(after)))/cleanScale, ...
                    mean(abs(oraclePred(after)-y(after)))/cleanScale, ...
                    ~isempty(pointHit),~isempty(stateHit), ...
                    delayValue(pointHit),delayValue(stateHit), ...
                    mean(pointFlags(1:max(1,onset-1))>0), ...
                    mean(stateFlags(1:max(1,onset-1))>0),countSwitches(stateFlags)};
            end
        end
    end
    T=cell2table(rows(1:row,:),'VariableNames', ...
        {'Dataset','Unit','Repeat','TargetSensor','FaultMode','SequenceLength', ...
        'FaultOnsetRecord','FullRelativeMAE','PointRelativeMAE', ...
        'StatefulRelativeMAE','OracleRelativeMAE','PointEventDetected', ...
        'StatefulEventDetected','PointDetectionDelay','StatefulDetectionDelay', ...
        'PointTriggerRate','StatefulTriggerRate','StatefulSwitchCount'});
    T.Dataset=string(T.Dataset); T.Unit=string(T.Unit);
    T.TargetSensor=string(T.TargetSensor); T.FaultMode=string(T.FaultMode);
    T.PointPreOnsetFalseRate=T.PointTriggerRate;
    T.StatefulPreOnsetFalseRate=T.StatefulTriggerRate;
end

function [Xvariant,quality,onset]=makeSequentialFault(P,mask,targetSensor,mode,seed)
    rng(seed,'twister'); index=find(mask); n=numel(index);
    [~,order]=sort(P.D.runTable.RunIndex(mask)); index=index(order);
    onset=max(2,min(n,round((0.20+0.25*rand)*n)));
    Xvariant=P.D.sensorX(mask,:); quality=P.cleanQuality(mask,:,:);
    endpointSeverity=0.5+1.5*rand; direction=2*(rand>0.5)-1;
    for orderedPosition=onset:n
        globalRecord=index(orderedPosition);
        localRow=find(find(mask)==globalRecord,1);
        progress=(orderedPosition-onset+1)/max(1,n-onset+1);
        x=P.rawSignals{globalRecord,targetSensor}; scale=std(x,0,'omitnan');
        if ~isfinite(scale)||scale<1e-10, scale=max(1e-6,0.01*max(abs(x))); end
        xc=x;
        switch mode
            case "LinearBiasDrift"
                xc=x+direction*endpointSeverity*progress*scale;
            case "GainDrift"
                gain=max(0.1,1+direction*(0.2+0.6*endpointSeverity/2)*progress);
                xc=x*gain;
            case "ColoredNoiseRamp"
                noise=filter(1,[1 -0.85],randn(size(x)));
                noise=noise/max(std(noise),1e-10);
                xc=x+scale*endpointSeverity*progress.*noise;
            case "IntermittentRecordDropout"
                if rand<(0.10+0.60*progress)
                    affected=false(numel(x),1); burstN=randi([2 6]);
                    fraction=0.05+0.35*progress;
                    lengthPer=max(1,round(fraction*numel(x)/burstN));
                    for b=1:burstN
                        start=randi(max(1,numel(x)-lengthPer+1));
                        affected(start:(start+lengthPer-1))=true;
                    end
                    xc(affected)=0;
                end
        end
        Xvariant(localRow,P.featureBlocks{targetSensor})=signalFeatures(xc);
        quality(localRow,targetSensor,:)=rawQualityDescriptors(xc);
    end
end

function flags=pointFlagsFromScores(scores)
    [maximum,flags]=max(scores,[],2); flags(maximum<=1)=0;
end

function flags=statefulFlags(scores,method,parameter,threshold,consecutive,releaseFraction)
    n=size(scores,1); nSensors=size(scores,2); state=zeros(n,nSensors);
    flags=zeros(n,1); active=false(1,nSensors); highCount=zeros(1,nSensors);
    lowCount=zeros(1,nSensors);
    for record=1:n
        if method=="EWMA"
            rho=parameter;
            if record==1, state(record,:)=scores(record,:);
            else
                state(record,:)=rho*state(record-1,:)+(1-rho)*scores(record,:);
            end
        else
            reference=parameter;
            if record==1
                state(record,:)=max(0,scores(record,:)-reference);
            else
                state(record,:)=max(0,state(record-1,:)+scores(record,:)-reference);
            end
        end
        high=state(record,:)>threshold;
        low=state(record,:)<releaseFraction*threshold;
        highCount(high)=highCount(high)+1; highCount(~high)=0;
        lowCount(low)=lowCount(low)+1; lowCount(~low)=0;
        active(highCount>=consecutive)=true;
        active(lowCount>=consecutive)=false;
        if any(active)
            candidate=state(record,:); candidate(~active)=-inf;
            [~,flags(record)]=max(candidate);
        end
    end
end

function [fullPred,dropPred]=branchPredictions(P,mask,policy,Xvariant)
    D=P.D; fullPred=predictRidge(policy.fullModel, ...
        [D.processX(mask,:),Xvariant,D.timeX(mask,:)]);
    dropPred=nan(sum(mask),3);
    for sensorIndex=1:3
        keep=policy.keepMasks{sensorIndex};
        dropPred(:,sensorIndex)=predictRidge(policy.dropModels{sensorIndex}, ...
            [D.processX(mask,:),Xvariant(:,keep),D.timeX(mask,:)]);
    end
end

function value=cleanFullMAE(P,mask,policy)
    prediction=predictRidge(policy.fullModel, ...
        [P.D.processX(mask,:),P.D.sensorX(mask,:),P.D.timeX(mask,:)]);
    value=max(mean(abs(prediction-P.D.y(mask))),1e-12);
end

function count=countSwitches(flags)
    count=sum(diff(flags)~=0);
end

function value=delayValue(index)
    if isempty(index), value=nan; else, value=index-1; end
end

function [p,W]=exactSignedRank(difference)
    difference=difference(isfinite(difference)&difference~=0);
    if isempty(difference), p=1; W=0; return; end
    [p,~,stats]=signrank(difference,0,'method','exact'); W=stats.signedrank;
end

function [lower,upper]=clusterBootstrapMeanCI(difference,B,seed)
    difference=difference(isfinite(difference)); n=numel(difference);
    stream=RandStream('mt19937ar','Seed',seed); boot=nan(B,1);
    for b=1:B, boot(b)=mean(difference(randi(stream,n,n,1))); end
    q=prctile(boot,[2.5 97.5]); lower=q(1); upper=q(2);
end

function value=datasetOffset(name)
    if name=="NUAA", value=10000000; else, value=20000000; end
end

function value=unitNumericId(unit)
    digits=regexp(char(unit),'\d+','match','once'); value=str2double(digits);
    if ~isfinite(value), value=sum(double(char(unit))); end
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
    else, flat=1; diffRms=0; end
    q=[rail,zero,flat,diffRms/max(std(x,0),1e-10)]; q(~isfinite(q))=0;
end

function f=signalFeatures(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), f=zeros(1,9); return; end
    mu=mean(x); rmsValue=sqrt(mean(x.^2)); sigma=std(x,0);
    peakToPeak=max(x)-min(x); med=median(x); madValue=median(abs(x-med));
    if sigma>max(eps(abs(mu)),1e-12)
        z=(x-mu)./sigma; skewValue=mean(z.^3); kurtValue=mean(z.^4);
    else, skewValue=0; kurtValue=0; end
    crest=max(abs(x))/max(rmsValue,eps);
    f=[mu,rmsValue,sigma,peakToPeak,skewValue,kurtValue,med,madValue,crest];
    f(~isfinite(f))=0;
end

function detector=fitQualityDetector(Xsensor,cleanQuality,featureBlocks, ...
        detectorFloor,detectorQuantile)
    selectedStats=[1 2 3 4 7 8 9]; detector.center=cell(1,3);
    detector.scale=cell(1,3); detector.threshold=zeros(1,3);
    for sensorIndex=1:3
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)), ...
            squeeze(cleanQuality(:,sensorIndex,:))];
        center=median(Q,1,'omitnan'); scale=1.4826*median(abs(Q-center),1,'omitnan');
        fallback=std(Q,0,1,'omitnan'); bad=~isfinite(scale)|scale<1e-9;
        scale(bad)=fallback(bad); scale(~isfinite(scale)|scale<1e-9)=1e-9;
        score=max(abs((Q-center)./scale),[],2);
        detector.center{sensorIndex}=center; detector.scale{sensorIndex}=scale;
        detector.threshold(sensorIndex)=max(detectorFloor,prctile(score,detectorQuantile));
    end
end

function score=qualityNormalizedScores(Xsensor,quality,detector,featureBlocks)
    selectedStats=[1 2 3 4 7 8 9]; n=size(Xsensor,1); score=zeros(n,3);
    for sensorIndex=1:3
        rawQ=squeeze(quality(:,sensorIndex,:));
        if n==1, rawQ=reshape(rawQ,1,[]); end
        Q=[Xsensor(:,featureBlocks{sensorIndex}(selectedStats)),rawQ];
        raw=max(abs((Q-detector.center{sensorIndex})./detector.scale{sensorIndex}),[],2);
        score(:,sensorIndex)=raw./detector.threshold(sensorIndex);
    end
end

function routed=routePredictions(fullPrediction,dropPrediction,flags)
    routed=fullPrediction;
    for sensorIndex=1:size(dropPrediction,2)
        rows=flags==sensorIndex; routed(rows)=dropPrediction(rows,sensorIndex);
    end
end

function lambda=groupedTuneLambda(X,y,groups,lambdaGrid)
    units=unique(groups,'stable'); scores=nan(numel(lambdaGrid),1);
    for lambdaIndex=1:numel(lambdaGrid)
        foldMAE=nan(numel(units),1);
        for unitIndex=1:numel(units)
            validation=groups==units(unitIndex); train=~validation;
            model=fitRidge(X(train,:),y(train),lambdaGrid(lambdaIndex));
            foldMAE(unitIndex)=mean(abs(predictRidge(model,X(validation,:))-y(validation)));
        end
        scores(lambdaIndex)=mean(foldMAE);
    end
    [~,best]=min(scores); lambda=lambdaGrid(best);
end

function model=fitRidge(X,y,lambda)
    X=double(X); y=double(y(:)); muX=mean(X,1,'omitnan'); muX(~isfinite(muX))=0;
    sigmaX=std(X,0,1,'omitnan'); sigmaX(~isfinite(sigmaX)|sigmaX<1e-12)=1;
    Z=(X-muX)./sigmaX; keep=std(Z,0,1)>1e-12; Z=Z(:,keep);
    muY=mean(y); beta=(Z'*Z+lambda*eye(size(Z,2)))\(Z'*(y-muY));
    model.muX=muX; model.sigmaX=sigmaX; model.keep=keep;
    model.beta=beta; model.muY=muY;
end

function yhat=predictRidge(model,X)
    Z=(double(X)-model.muX)./model.sigmaX;
    yhat=model.muY+Z(:,model.keep)*model.beta;
end
