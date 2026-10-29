function IsInT = CheckIsInT(Point2DXT,Point2DYT,Iris,IrisC,dens,fast,NofProc,verbose)

% NofProc = 2;  % Number of available processes

if nargin<8 || isempty(verbose)
   verbose=0;
end


if verbose>0
   fprintf('\n');
end

IsInT = zeros(size(Point2DXT));

if fast == 0
  for i1 = 1:size(Point2DXT,1)
    for i2 = 1:size(Point2DXT,2)

    IsInT(i1,i2) = CheckIsIn(Point2DXT(i1,i2),Point2DYT(i1,i2),Iris,IrisC,dens,0);

    if verbose>0
      fprintf('.');
    end

    end
  end
elseif fast==1

  Iris2DX = Iris(2,:);
  Iris2DY = Iris(3,:);

  Iris2DCX = IrisC(2);
  Iris2DCY = IrisC(3);

  ArcDegT = zeros(numel(Iris2DX),1);
  for i=1:numel(Iris2DX)
   ArcDegT(i) = ArcDeg(Iris2DX(i)  -Iris2DCX,Iris2DY(i)  -Iris2DCY);
  end

  for i1 = 1:size(Point2DXT,1)
    for i2 = 1:size(Point2DXT,2)

    IsInT(i1,i2) = ...
    CheckIsInF(Point2DXT(i1,i2),Point2DYT(i1,i2),Iris2DX,Iris2DY,Iris2DCX,Iris2DCY,dens,ArcDegT,0);

    end
  end

else      % fast == 2

  Iris2DX = Iris(2,:);
  Iris2DY = Iris(3,:);

  Iris2DCX = IrisC(2);
  Iris2DCY = IrisC(3);

  ArcDegT = zeros(numel(Iris2DX),1);
  for i=1:numel(Iris2DX)
   ArcDegT(i) = ArcDeg(Iris2DX(i)  -Iris2DCX,Iris2DY(i)  -Iris2DCY);
  end


  IsInT = ...
    pararrayfun(NofProc,@(xx,yy) CheckIsInF(xx,yy,Iris2DX,Iris2DY,Iris2DCX,Iris2DCY,dens,ArcDegT,0),Point2DXT,Point2DYT);


end
%

if verbose>0
   fprintf('\n');
end

end
