% =========================================================================
% NOTE. Lines marked "[mod]" differ from the original script, and the
% descriptive header below was reworded.
%
% The header comment that follows states
%     r = 1 / (DensCCD * 5)
% and calls it the sensor resolution. That expression is the pitch of the
% INTERNAL RENDERING GRID, which IrisC.m builds at 5*DensCCD lines/mm
% (IrisC.m line 45). reduce5.m then averages 5x5 blocks of that grid to form
% one sensor pixel, and the centroid the model reports (xcMr, ycMr, columns
% 5 and 6) is computed on the reduced grid. The SENSOR pixel pitch is
% therefore
%     r_sensor = 1 / DensCCD              [mm/line]
% five times coarser: 1.00 mm/line at DensCCD = 1, where the 5 mm pupil spans
% about 5 sensor pixels rather than 25. Measured directly from this code, the
% reduced grid spacing at DensCCD = 1 is 1.025 mm.
%
% The variable rT saved below is left exactly as the original wrote it, so
% Result15.mat is unchanged; rSensorT is added next to it. The Python
% implementation uses r_sensor.
% =========================================================================

% iris15_resolution.m
% -------------------------------------------------------------------------
% Sweep of CMOS sensor RESOLUTION.
%
% This script computes the standard deviation of the pupil's centre of
% mass s against the CMOS sensor resolution r.
% It mirrors iris14.m (population-based pupil-shape model with random
% harmonic distortions) but, instead of sweeping the noise level, it sweeps
% the sensor resolution DensCCD at a fixed, low noise level.
%
% Resolution convention (see IrisC.m):
%   DensCCD_internal = DensCCD * 5         [lines/mm]
%   r = 1 / DensCCD_internal               [mm/line]
% e.g. DensCCD = 1   ->  5 lines/mm  ->  r = 0.20 mm/line
%      DensCCD = 5   -> 25 lines/mm  ->  r = 0.04 mm/line
%
% Output: Result15.mat with the SAME ResTT layout as Result13/14:
%   ResTT(iRes, iAngle, 1, :) = std  over Rep trials   (columns 1..6)
%   ResTT(iRes, iAngle, 2, :) = mean over Rep trials
%   columns: [xcR ycR xcM ycM xcMr ycMr]  (col 5,6 = low-res + noise = real)
%
% Author: Roman Szewczyk / Anna Ostaszewska-Lizewska (2025), MIT license
% -------------------------------------------------------------------------

clear all
clc

pkg load parallel

IrisRotVers = [0;1;2];          % rotation axis (same as iris14)

fast    = 2;                    % 2 = pararrayfun (parallel). Set 1 for serial if unstable.
NofProc = 30;                   % number of parallel processes (used only if fast=2)
Rep     = 3600;                 % Monte-Carlo trials per operating point
verbose = 0;

% --- swept parameter: sensor resolution -------------------------------
% NOTE: the per-evaluation cost grows ~quadratically with DensCCD (finer
% grid), so DensCCD up to ~2 keeps the run tractable; values >=3 are very
% slow/memory-heavy.
DensCCDT = [0.3 0.4 0.5 0.7 1.0 1.5 2.0];            % -> r below
rT       = 1 ./ (DensCCDT .* 5);                     % [mm/line], for reference
rSensorT = 1 ./ DensCCDT;                            % [mod] sensor pixel pitch
% r = [0.667 0.500 0.400 0.286 0.200 0.133 0.100] mm/line; baseline = 0.200

% --- fixed noise level (low, so the resolution effect is visible) -----
NoiseL = 0.10;

% --- gaze angles (same grid as iris14) --------------------------------
IrisRotDegT = -60:15:60;

fprintf('\n\n Resolution sweep started: \n');
fprintf(' DensCCDT     = %s\n', mat2str(DensCCDT));
fprintf(' rendering pitch [mm]      = %s\n', mat2str(round(rT*1e4)/1e4));
fprintf(' SENSOR r [mm/line] = %s\n', mat2str(round(rSensorT*1e4)/1e4));
fprintf(' NoiseL fixed = %.3f,  Rep = %d\n\n', NoiseL, Rep);

ResTT = zeros(numel(DensCCDT), numel(IrisRotDegT), 2, 6);

for k = 1:numel(DensCCDT)

  DensCCD = DensCCDT(k);

  for j = 1:numel(IrisRotDegT)

    IrisRotDeg = IrisRotDegT(j);
    ResT = [];

    tic();

    for i = 1:Rep

      % population-based random pupil-shape distortion (as in iris14.m)
      [IrisDistAmp, IrisDistPhase] = PopulationDistortion();

      [xcR,ycR,xcM,ycM,xcMr,ycMr] = IrisC(IrisRotVers,IrisRotDeg,IrisDistAmp, ...
                                          IrisDistPhase,DensCCD,NoiseL, ...
                                          fast,NofProc,verbose);

      ResT = [ResT; xcR,ycR,xcM,ycM,xcMr,ycMr];

    end
    %

    t = toc();

    % [mod] Drop trials in which the thresholded low-resolution image retained no
    % foreground pixel, so that IrisC returned NaN for the centroid (columns 5 and 6);
    % otherwise std turns the whole operating point into NaN.
    nBad = sum(any(isnan(ResT), 2));
    ResT = ResT(~any(isnan(ResT), 2), :);
    if nBad > 0
      fprintf('  [%d of %d trials discarded: no foreground after threshold]\n', nBad, Rep);
    end

    ResTT(k,j,1,:) = std(ResT);
    ResTT(k,j,2,:) = mean(ResT);

    fprintf('DensCCD = %1.2f (sensor r = %1.4f mm/line), angle = %1.1f deg done in %1.1fs.\n', ...
            DensCCD, rSensorT(k), IrisRotDeg, t);   % [mod] was rT(k), the rendering pitch

    save -v7 Result15_temp.mat IrisRotVers DensCCDT rT rSensorT NoiseL ...
                        fast NofProc verbose Rep ...
                        IrisRotDegT ResT ResTT

  end
  %
end
%

fprintf('\n\n*** Resolution sweep done. ***\n\n');

save -v7 Result15.mat IrisRotVers DensCCDT rT rSensorT NoiseL ...
                      fast NofProc verbose Rep ...
                      IrisRotDegT ResT ResTT
