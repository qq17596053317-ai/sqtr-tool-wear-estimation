clear; clc;

paperRoot=fileparts(mfilename('fullpath'));
dataRoot=string(getenv('SQTR_RAW_DATA_DIR'));
if strlength(dataRoot)==0, dataRoot="C:\Users\66485\Desktop\论文"; end
resultDir=string(getenv('SQTR_WAVEFORM_AUDIT_RESULT_DIR'));
if strlength(resultDir)==0
    resultDir=fullfile(paperRoot,'targeted_reviewer_experiments', ...
        'waveform_feature_reconstruction_audit','results');
end
if ~exist(resultDir,'dir'), mkdir(resultDir); end

S=load(fullfile(paperRoot,'external_validation_results', ...
    'external_validation_complete_results.mat'),'nuaa','phm');
specifications={ ...
    S.nuaa,fullfile(dataRoot,'nuaa_orthogonal_bundle_high_resolution.csv'),"NUAA"; ...
    S.phm,fullfile(dataRoot,'phm2010_bundle_high_resolution.csv'),"PHM2010"};
statNames=["Mean","RMS","Std","PeakToPeak","Skewness", ...
    "Kurtosis","Median","MAD","CrestFactor"];

detailParts=cell(2,1); sensorRows={}; sensorRow=0; datasetRows={};
for datasetIndex=1:2
    D=specifications{datasetIndex,1}; filePath=specifications{datasetIndex,2};
    datasetName=specifications{datasetIndex,3};
    fprintf('Reconstructing %s features from high-resolution signals...\n',datasetName);
    T=readtable(filePath,'VariableNamingRule','preserve');
    groupId=makeRawGroupId(T,datasetName);
    reconstructed=nan(size(D.sensorX));
    sampleCounts=zeros(numel(D.y),1);
    for recordIndex=1:numel(D.y)
        rows=groupId==D.runTable.SourceGroup(recordIndex);
        assert(any(rows),'No signal samples for %s record %d.',datasetName,recordIndex);
        sampleCounts(recordIndex)=sum(rows);
        for sensorIndex=1:numel(D.sensorNames)
            block=(sensorIndex-1)*9+(1:9);
            reconstructed(recordIndex,block)=signalFeatures( ...
                T.(char(D.sensorNames(sensorIndex)))(rows));
        end
    end
    difference=reconstructed-D.sensorX;
    scale=max(std(D.sensorX,0,1),1e-12);
    standardised=abs(difference)./scale;
    [recordGrid,featureGrid]=ndgrid((1:numel(D.y))',1:size(D.sensorX,2));
    sensorIndex=ceil(featureGrid(:)/9); statIndex=mod(featureGrid(:)-1,9)+1;
    detailParts{datasetIndex}=table(repmat(datasetName,numel(difference),1), ...
        string(D.experiment(recordGrid(:))),D.runTable.RunIndex(recordGrid(:)), ...
        D.runTable.SourceGroup(recordGrid(:)),sampleCounts(recordGrid(:)), ...
        string(D.sensorNames(sensorIndex))',statNames(statIndex)', ...
        D.sensorX(sub2ind(size(D.sensorX),recordGrid(:),featureGrid(:))), ...
        reconstructed(sub2ind(size(reconstructed),recordGrid(:),featureGrid(:))), ...
        difference(:),abs(difference(:)),standardised(:), ...
        'VariableNames',{'Dataset','Unit','RunIndex','SourceGroup','SignalSampleN', ...
        'Sensor','Statistic','StoredValue','ReconstructedValue','Difference', ...
        'AbsoluteDifference','StandardisedAbsoluteDifference'});
    for s=1:numel(D.sensorNames)
        block=(s-1)*9+(1:9); values=abs(difference(:,block)); z=standardised(:,block);
        sensorRow=sensorRow+1;
        sensorRows(sensorRow,:)={datasetName,D.sensorNames(s),numel(values), ...
            median(values,'all'),max(values,[],'all'),median(z,'all'),max(z,[],'all')};
    end
    datasetRows(datasetIndex,:)={datasetName,numel(D.y),numel(D.sensorNames), ...
        sum(sampleCounts),median(abs(difference),'all'),max(abs(difference),[],'all'), ...
        median(standardised,'all'),max(standardised,[],'all'), ...
        mean(abs(difference(:))<=1e-12)};
end

detailTable=vertcat(detailParts{:});
sensorTable=cell2table(sensorRows,'VariableNames',{'Dataset','Sensor', ...
    'FeatureValueN','MedianAbsoluteDifference','MaximumAbsoluteDifference', ...
    'MedianStandardisedAbsoluteDifference','MaximumStandardisedAbsoluteDifference'});
datasetTable=cell2table(datasetRows,'VariableNames',{'Dataset','RecordN','SensorN', ...
    'HighResolutionSampleN','MedianAbsoluteDifference','MaximumAbsoluteDifference', ...
    'MedianStandardisedAbsoluteDifference','MaximumStandardisedAbsoluteDifference', ...
    'FractionWithin1eMinus12'});
sensorTable.Dataset=string(sensorTable.Dataset); sensorTable.Sensor=string(sensorTable.Sensor);
datasetTable.Dataset=string(datasetTable.Dataset);

writetable(datasetTable,fullfile(resultDir,'waveform_reconstruction_dataset_summary.csv'));
writetable(sensorTable,fullfile(resultDir,'waveform_reconstruction_sensor_summary.csv'));
writetable(detailTable,fullfile(resultDir,'waveform_reconstruction_detail.csv'));
save(fullfile(resultDir,'waveform_reconstruction_audit.mat'), ...
    'datasetTable','sensorTable','detailTable','-v7.3');
disp(datasetTable); fprintf('Saved to %s\n',resultDir);

function groupId=makeRawGroupId(T,datasetName)
    tag=string(T.experiment_tag); timestamp=double(T.timestamp);
    if datasetName=="NUAA"
        [groupId,~,~]=findgroups(tag,double(T.experiment_csv_n));
    else
        groupId=zeros(height(T),1); nextGroup=0; tags=unique(tag,'stable');
        for i=1:numel(tags)
            rows=find(tag==tags(i)); [sortedTimestamp,order]=sort(timestamp(rows));
            rows=rows(order); increment=diff(sortedTimestamp);
            cadence=median(increment(increment>0),'omitnan');
            assert(isfinite(cadence) && cadence>0,'Invalid timestamp cadence.');
            localRun=cumsum([true;increment<0.1*cadence]);
            groupId(rows)=nextGroup+localRun; nextGroup=nextGroup+max(localRun);
        end
    end
end

function f=signalFeatures(x)
    x=double(x(:)); x=x(isfinite(x));
    if isempty(x), f=zeros(1,9); return; end
    mu=mean(x); rmsValue=sqrt(mean(x.^2));
    if numel(x)>1, sigma=std(x,0); else, sigma=0; end
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
