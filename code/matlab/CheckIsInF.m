function IsIn = CheckIsInF(Point2DX,Point2DY,Iris2DX,Iris2DY,Iris2DCX,Iris2DCY,dens,ArcDegT,verbose);

%Iris2DX = Iris(2,:);
%Iris2DY = Iris(3,:);
%
%Iris2DCX = IrisC(2);
%Iris2DCY = IrisC(3);

semafor = 0;
i=1;

fic = ArcDeg(Point2DX-Iris2DCX,Point2DY-Iris2DCY);

while ((semafor==0)&&(i<numel(Iris2DX)))

  fi1 = ArcDegT(i);   %(Iris2DX(i)  -Iris2DCX,Iris2DY(i)  -Iris2DCY);
  fi2 = ArcDegT(i+1); %(Iris2DX(i+1)-Iris2DCX,Iris2DY(i+1)-Iris2DCY);

  if fi1-fi2>3.*360./dens     % avoid zero crossing
    fi2=360+fi2;
  end

  if fi2-fi1>3.*360./dens     % avoid zero crossing
    fi1=360+fi1;
  end

  if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
    semafor = 1;
  end

  if ((fi1>360) || (fi2>360))
    fi1 = fi1-360;
    fi2 = fi2-360;
    fic = fic-360;

    if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
      semafor = 1;
    else
      fic = fic+360;
    end

    if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
      semafor = 1;
    end

  end

  i=i+1;
end
%
i=i-1;

if semafor == 0
  % No bracketing boundary segment found for this grid point.
  % Treat the point as lying OUTSIDE the pupil shape. Assigning the output
  % here is essential: returning without setting IsIn breaks pararrayfun
  % ("could not receive result") in the parallel (fast=2) path.
  if verbose>0
    fprintf('\n\n*** Warning: no angle detection. fic = %f x = %f, y = %f \n\n',fic, Point2DX,Point2DY);
  end
  IsIn = 0;
  return;
end
%

% Check if within linear interpolation of the shape

Tabl1 = [fi1; fic; fi2];                        % Create the angle-length table
Tabl2 = [VectL(Iris2DX(i)  -Iris2DCX,Iris2DY(i)  -Iris2DCY); ...
         VectL(Point2DX-Iris2DCX,Point2DY-Iris2DCY);
         VectL(Iris2DX(i+1)-Iris2DCX,Iris2DY(i+1)-Iris2DCY)];
Tabl = [Tabl1 Tabl2];


Tabl = sortrows(Tabl,1);          % sort the angle-length table

LLimit = Tabl(1,2)+...
        ((Tabl(2,1)-Tabl(1,1))./(Tabl(3,1)-Tabl(1,1))).* ...
        (Tabl(3,2)-Tabl(1,2));

if LLimit>Tabl(2,2)
   IsIn = 1;
else
   IsIn = 0;
end
%

end
