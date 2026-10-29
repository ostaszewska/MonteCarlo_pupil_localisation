clear all
clc

pkg load parallel

IrisRotVers = [0;1;2];
% IrisRotDeg = 25;

% Iris distorsions
% harmonic amp (m)      2       3      4      5      6      7
IrisDistAmp =       [ 0.0e-3 0.0e-3 0.0e-3 0.0e-3 0.0e-3 0.0e-3];
% harmonic phase (deg)  2       3      4      5      6      7
IrisDistPhase =     [  0.0     0.0    0.0    0.0    0.0    0.0];

DensCCD = 1;     % in lines per mm
% NoiseL = 0.1;

fast = 2;
NofProc = 30;

Rep = 3600;

verbose = 0;

ResTT = [];

fprintf('\n\n Process started: \n');

IrisRotDegT = -60:15:60;
NoiseLT = 0.05:0.05:0.5;

ResTT = zeros(numel(NoiseLT),numel(IrisRotDegT),2,6);

for k=1:numel(NoiseLT)

  NoiseL = NoiseLT(k);

  for j=1:numel(IrisRotDegT)

    IrisRotDeg = IrisRotDegT(j);
    ResT = [];

    tic();

    for i=1:Rep

      [IrisDistAmp, IrisDistPhase] = PopulationDistortion();


      [xcR,ycR,xcM,ycM,xcMr,ycMr] = IrisC(IrisRotVers,IrisRotDeg,IrisDistAmp,IrisDistPhase,DensCCD,NoiseL,...
                                                   fast,NofProc,verbose);

      ResT = [ResT; xcR,ycR,xcM,ycM,xcMr,ycMr];

    end
    %

  t=toc();

  % [mod] Drop trials in which the thresholded low-resolution image retained no
  % foreground pixel, so that IrisC returned NaN for the centroid (columns 5 and 6);
  % otherwise std turns the whole operating point into NaN. About 0.2% of trials at
  % DensCCD = 1 and low noise.
  nBad = sum(any(isnan(ResT), 2));
  ResT = ResT(~any(isnan(ResT), 2), :);
  if nBad > 0
    fprintf('  [%d of %d trials discarded: no foreground after threshold]\n', nBad, Rep);
  end

  ResTT(k,j,1,:) = std(ResT);
  ResTT(k,j,2,:) = mean(ResT);

  fprintf('NoiseLT = %1.2f, IrisRotDegT = %1.1f done in %1.1fs.\n',NoiseL,IrisRotDeg,t);

  save -v7 Result13_temp.mat IrisRotVers IrisRotDeg IrisDistAmp IrisDistPhase DensCCD NoiseLT ...
                      fast NofProc verbose ...
                      IrisRotDegT ResT ResTT

  end
%
end%

fprintf('\n\n*** Everything done. ***\n\n');

save -v7 Result14.mat IrisRotVers IrisRotDeg IrisDistAmp IrisDistPhase DensCCD NoiseLT ...
                      fast NofProc verbose ...
                      IrisRotDegT ResT ResTT


