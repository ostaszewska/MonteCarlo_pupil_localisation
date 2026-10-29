function [xcR,ycR,xcM,ycM,xcMr,ycMr] = IrisC(IrisRotVers,IrisRotDeg,IrisDistAmp,IrisDistPhase,DensCCD,NoiseL,fast,NofProc,verbose)

% environmental parameters

% verbose = 2;

if verbose==2
  set (0, "DefaultAxesFontName", "Arial")
  set(0, "DefaultAxesFontSize", 12)
end
%

if verbose==0
  warning('off','all');
end
%

% fast = 1;  % fast = 0 - reference with for-for loop
             % fast = 1 - faster with data in the table
             % fast = 2 - pararrayfun from parallel package

% NofProc = 2;   % the number of available processes for fast=2


% Model parameters

dens = 100;               % density of points on plot, no. per full rotation

EyeBallDiameter = 0.025;   % Diameter of the eye ball
IrisDiameter = 0.005;     % Diameter of the iris

% Iris distorsions
% harmonic amp (m)      2       3      4      5      6      7
% IrisDistAmp =       [ 0.0e-3 1.5e-3 0.0e-3 0.0e-3 0.0e-3 0.0e-3];
% harmonic phase (deg)  2       3      4      5      6      7
% IrisDistPhase =     [  0.0     0.0    0.0    0.0    0.0    0.0];

% IrisRotDeg = 45;          % rotation of the iris position
IrisRotVers = VectNorm(IrisRotVers);  % versor for the rotation of the iris position

IrisX = sqrt((EyeBallDiameter./2).^2-(IrisDiameter./2).^2);   % X component of iris center
IrisC = [IrisX,0,0];                                          % position  of iris center

% DensCCD = 1;     % in lines per mm
DensCCD = DensCCD .*5;

margP = 0.5;    % margin parameter

% NoiseL = 0.1;   % level of normal noise

% Prepare data for eye ball sphere plot

EyeBall1 = [];

for i=0:dens

  fi = i./dens.*360;
  EyeBall1 = [EyeBall1 RotVectX([0;0;1].*EyeBallDiameter./2,fi)];

end
%

EyeBall2 = [];

for i=0:dens

  fi = i./dens.*360;
  EyeBall2 = [EyeBall2 RotVectZ([1;0;0].*EyeBallDiameter./2,fi)];

end
%

EyeBall3 = [];

for i=0:dens

  fi = i./dens.*360;
  EyeBall3 = [EyeBall3 RotVectY([0;0;1].*EyeBallDiameter./2,fi)];

end
%

% prepare data for iris plot

Iris = [];

for i=0:dens                                            % create iris

  fi = i./dens.*360;

  IrisDiameter_ = IrisDiameter;
  for j=1:numel(IrisDistAmp)
    IrisDiameter_ = IrisDiameter_ + ...
                    IrisDistAmp(j).*sin((fi.*(j+1)+IrisDistPhase(j)).*pi./180);
  end

  Iris0 = RotVectX([IrisX;0;IrisDiameter_./2],fi);       % iris circle
  IrisR = RotVectVers(Iris0, IrisRotVers, IrisRotDeg);  % iris rotation
  Iris = [Iris IrisR];

end
%
IrisC = RotVectVers(IrisC, IrisRotVers, IrisRotDeg);    % iris center

if verbose==2

figure (1,'position',[100,200,600,600])   % figure(1, 'position',[startx,starty,width,height]);

  plot3(EyeBall1(1,:),EyeBall1(2,:),EyeBall1(3,:),':r');
  hold on;
  plot3(EyeBall2(1,:),EyeBall2(2,:),EyeBall2(3,:),':m');
  plot3(EyeBall3(1,:),EyeBall3(2,:),EyeBall3(3,:),':b');
  plot3(Iris(1,:),Iris(2,:),Iris(3,:),'-k');
  plot3(IrisC(1,:),IrisC(2,:),IrisC(3,:),'xk');
  hold off;

  grid;
  xlabel('X');
  ylabel('Y');
  zlabel('Z');
  view (90,0);
end
%


% Checking if the points are inside of modified eclipse shape
margX  = margP.*(max(Iris(2,:))-min(Iris(2,:)));
startX = min(Iris(2,:)) - margX.*(rand(1,1)+1);
stopX  = max(Iris(2,:)) + margX.*(rand(1,1)+1);

margY  = margP.*(max(Iris(3,:))-min(Iris(3,:)));
startY = min(Iris(3,:)) - margY.*(rand(1,1)+1);
stopY  = max(Iris(3,:)) + margY.*(rand(1,1)+1);

x = linspace(startX,stopX,floor(1e3.*DensCCD.*(stopX-startX)));
y = linspace(startY,stopY,floor(1e3.*DensCCD.*(stopY-startY)));

[Point2DXT,Point2DYT] = meshgrid(x,y);

if verbose>0
  tic;
end
%

IsInT1 = CheckIsInT(Point2DXT,Point2DYT,Iris,IrisC,dens,fast,NofProc,verbose);

if verbose>0
  toc;
end
%

if verbose==2
  figure (4,'position',[100,200,600,600])   % figure(1, 'position',[startx,starty,width,height]);
  pcolor(Point2DXT,Point2DYT,IsInT1);
  xlim([startX,stopX]);
  ylim([startY,stopY]);
end
%

[Point2DXTr,Point2DYTr,IsInT1r] = reduce5 (Point2DXT,Point2DYT,IsInT1);

if verbose==2
  figure (6,'position',[100,200,600,600])   % figure(1, 'position',[startx,starty,width,height]);
  pcolor(Point2DXTr,Point2DYTr,IsInT1r);
  xlim([startX,stopX]);
  ylim([startY,stopY]);
end
%

IsInT1r = IsInT1r+randn(size(IsInT1r)).*NoiseL;   % add noise

IsInT1r(IsInT1r<0.5)  =0;
IsInT1r(IsInT1r>=0.5) =1;         % apply treshold

if verbose==2
  figure (7,'position',[100,200,600,600])   % figure(1, 'position',[startx,starty,width,height]);
  pcolor(Point2DXTr,Point2DYTr,IsInT1r);
  xlim([startX,stopX]);
  ylim([startY,stopY]);
end
%

xc = sum(sum(Point2DXT.*IsInT1))./sum(sum(IsInT1));
yc = sum(sum(Point2DYT.*IsInT1))./sum(sum(IsInT1));     % calculate optimised sum

xcr = sum(sum(Point2DXTr.*IsInT1r))./sum(sum(IsInT1r));
ycr = sum(sum(Point2DYTr.*IsInT1r))./sum(sum(IsInT1r)); % calculate real sum

if verbose>0
  fprintf('\n');
  fprintf('The centre of mass:\n');
  fprintf('Generated   : %1.6e %1.6e \n',IrisC(2),IrisC(3));
  fprintf('Calculated  : %1.6e %1.6e as the center of mass \n',xc,yc);
  fprintf('Calculated r: %1.6e %1.6e as the center of mass \n\n',xcr,ycr);
end
%

xcR = IrisC(2);
ycR = IrisC(3);

xcM = xc;
ycM = yc;

xcMr = xcr;
ycMr = ycr;

end % of function
