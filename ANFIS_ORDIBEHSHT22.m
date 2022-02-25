






clc
close all
clear TrainInputs
clear TrainTargets
clear TestOutputs
clear TestInputs
clear TestTargets
%+++++++++++++++++++++++       ANFIS     ++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Inputs=InputAA;
    Targets=TargetAA;
    
    nSample=size(Inputs,1);
    
    S=randperm(nSample);
    Inputs=Inputs(S,:);
    Targets=Targets(S,:);
    
   C=horzcat(S',Inputs);
    [mmm nnn]=size(C);
    
  
    
    % Train Data
    pTrain=0.9;
    nTrain=round(pTrain*nSample);
    TrainInputs=Inputs(1:nTrain,:);
    TrainTargets=Targets(1:nTrain,:);
    
    
    
    % Test Data
    TestInputs=Inputs(nTrain+1:end,:);
    TestTargets=Targets(nTrain+1:end,:);

   
    
    
nData=size(Inputs,1);
nInputs=size(Inputs,2)
nOutputs=size(Targets,2)

%% Design ANFIS

Option{1}='Grid Part. (genfis1)';
Option{2}='Sub. Clustering (genfis2)';
Option{3}='FCM (genfis3)';

ANSWER=questdlg('Select FIS Generation Approach:',...
                'Select GENFIS',...
                Option{1},Option{2},Option{3},...
                Option{3});
pause(0.1);

switch ANSWER
    case Option{1}
        Prompt={'Number of MFs','Input MF Type:','Output MF Type:'};
        Title='Enter genfis1 parameters';
        DefaultValues={'5','gaussmf','linear'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        nMFs=str2num(PARAMS{1}); %#ok
        InputMF=PARAMS{2};
        OutputMF=PARAMS{3};
        
        fis=cell(nOutputs,1);
        for j=1:nOutputs
            fis{j}=genfis1([TrainInputs TrainTargets(:,j)],nMFs,InputMF,OutputMF);
        end

    case Option{2}
        Prompt={'Influence Radius:'};
        Title='Enter genfis2 parameters';
        DefaultValues={'0.2'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        Radius=str2num(PARAMS{1}); %#ok
        
        fis=cell(nOutputs,1);
        for j=1:nOutputs
            fis{j}=genfis2(TrainInputs,TrainTargets(:,j),Radius);
        end
        
    case Option{3}
        Prompt={'Number fo Clusters:',...
                'Partition Matrix Exponent:',...
                'Maximum Number of Iterations:',...
                'Minimum Improvemnet:'};
        Title='Enter genfis3 parameters';
        DefaultValues={'10','2','100','1e-5'};
        
        PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
        pause(0.1);

        nCluster=str2num(PARAMS{1}); %#ok
        Exponent=str2num(PARAMS{2}); %#ok
        MaxIt=str2num(PARAMS{3}); %#ok
        MinImprovment=str2num(PARAMS{4}); %#ok
        DisplayInfo=1;
        FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
        
        fis=cell(nOutputs,1);
        for j=1:nOutputs
            fis{j}=genfis3(TrainInputs,TrainTargets(:,j),'sugeno',nCluster,FCMOptions);
        end
end

Prompt={'Maximum Number of Epochs:',...
        'Error Goal:',...
        'Initial Step Size:',...
        'Step Size Decrease Rate:',...
        'Step Size Increase Rate:'};
Title='Enter genfis3 parameters';
DefaultValues={'100','0','0.01','0.9','1.1'};

PARAMS=inputdlg(Prompt,Title,1,DefaultValues);
pause(0.1);

MaxEpoch=str2num(PARAMS{1});                %#ok
ErrorGoal=str2num(PARAMS{2});               %#ok
InitialStepSize=str2num(PARAMS{3});         %#ok
StepSizeDecreaseRate=str2num(PARAMS{4});    %#ok
StepSizeIncreaseRate=str2num(PARAMS{5});    %#ok
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid

for j=1:nOutputs
    fis{j}=anfis([TrainInputs TrainTargets(:,j)],fis{j},TrainOptions,DisplayOptions,[],OptimizationMethod);
end


%% Apply ANFIS to Train Data

TrainOutputs=zeros(size(TrainTargets));
for j=1:nOutputs
    TrainOutputs(:,j)=evalfis(TrainInputs,fis{j});
end

TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

for j=1:nOutputs
    Title=['Train Data - Output #' num2str(j)];
    
    figure;
    PlotResults(TrainTargets(:,j),TrainOutputs(:,j),Title);

    figure;
    plotregression(TrainTargets(:,j),TrainOutputs(:,j),Title);
    set(gcf,'Toolbar','figure');
end



%% Apply ANFIS to Test Data
TestOutputs=zeros(size(TestTargets));
nOutputs=size(TestTargets,2)
for j=1:nOutputs
    TestOutputs(:,j)=evalfis(TestInputs,fis{j});
end

TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

for j=1:nOutputs
    Title=['Test Data - Output #' num2str(j)];
    
    figure;
    PlotResults(TestTargets(:,j),TestOutputs(:,j),Title);
    figure
   
plotregression(TestTargets(:,j),TestOutputs(:,j),Title);
    set(gcf,'Toolbar','figure');
    
end
%% Test for an Example Path

