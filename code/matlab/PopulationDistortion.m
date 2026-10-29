function [IrisDistAmp, IrisDistPhase] = PopulationDistortion()
% POPULATIONDISTORTION  Random population pupil-shape distortion, after Wyatt (1995).
%
%   [IrisDistAmp, IrisDistPhase] = PopulationDistortion()
%
%   IrisDistAmp   amplitude of the modulation of the pupil DIAMETER, in metres,
%                 for harmonics 2..8, as consumed by IrisC.m line 93.
%   IrisDistPhase phase of each harmonic, in degrees.
%
%   Wyatt represents the pupil margin as a circular Fourier series (his eq. 1)
%
%       R(theta) = r_ave + sum_{n>=2} r_n cos n(theta - phi_n)
%
%   and characterises shape by the noncircularity (his eq. 3)
%
%       NC^2 = 1/2 sum_{n>=2} (r_n / r_ave)^2 ,
%
%   which is normalised to the AVERAGE RADIUS and therefore dimensionless. He
%   measured NC = 0.0176 +- 0.0062 in the dark and 0.0166 +- 0.0041 in the light
%   over 23 subjects, with the best-fit ellipse contributing 0.596 of the energy
%   in the dark, and pupil diameters of 4.93 mm (dark) and 3.09 mm (light).
%
%   The harmonic amplitudes below are read off his Fig. 4, whose x-axis is
%   10^4 x (r_n / r_ave). A normalised radius amplitude converts to a diameter
%   modulation as
%
%       dist_amp = (r_n / r_ave) * 2 * r_ave = (r_n / r_ave) * IrisDiameter .
%
%   Earlier versions of iris14.m and iris15_resolution.m applied the seven
%   numbers directly as metres, which inflates every amplitude about 20-fold and
%   raises NC from 0.0176 to 0.296; this function applies the conversion.
%
%   The Python reference implementation is iris_model.population_distortion.

  IrisDiameter = 0.005;          % m, must match IrisC.m line 30

  % Wyatt Fig. 4, harmonics 2..8, in units of 1e-4 x (r_n / r_ave)
  base   = [150 100 50 50 20 20 10] * 1e-4;
  spread = [ 75  50 25 25 10 10  5] * 1e-4;

  % One overall factor puts the mean noncircularity of these seven numbers,
  % 0.014836 as read, onto Wyatt's measured mean. The harmonic energy shares
  % are untouched. Condition 'dark' is matched to the 5 mm model pupil.
  NC_AS_READ = 0.014836;
  NC_WYATT   = 0.0176;           % dark; use 0.0166 for the light condition
  k = NC_WYATT / NC_AS_READ;

  IrisDistPhase = rand(1,7).*358 + 1;

  l = randn(1,7);
  l(l < -2) = -2;
  l(l >  2) =  2;

  IrisDistAmp = (base + l .* spread) * k * IrisDiameter;
end
