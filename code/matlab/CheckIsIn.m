function IsIn = CheckIsIn(Point2DX,Point2DY,Iris,IrisC,dens,verbose)

Iris2DX = Iris(2,:);
Iris2DY = Iris(3,:);

Iris2DCX = IrisC(2);
Iris2DCY = IrisC(3);

semafor = 0;
i=1;

fic = ArcDeg(Point2DX-Iris2DCX,Point2DY-Iris2DCY);

while ((semafor==0)&&(i<numel(Iris2DX)))

  fi1 = ArcDeg(Iris2DX(i)  -Iris2DCX,Iris2DY(i)  -Iris2DCY);
  fi2 = ArcDeg(Iris2DX(i+1)-Iris2DCX,Iris2DY(i+1)-Iris2DCY);

  if fi1-fi2>3.*360./dens     % avoid zero crossing
    fi2=360+fi2;
  end

  if fi2-fi1>3.*360./dens     % avoid zero crossing
    fi1=360+fi1;
  end

  if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
    semafor = 1;
    if verbose>0
      fprintf('%f %f %f\n',fi1,fic,fi2);
    end
  end

  if ((fi1>360) || (fi2>360))
    fi1 = fi1-360;
    fi2 = fi2-360;
    fic = fic-360;

    if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
      semafor = 1;
      if verbose>0
        fprintf('It is here (b)! %f %f %f\n',fi1,fic,fi2);
      end
    else
      if verbose>0
        fprintf('(b) %f %f %f\n',fi1,fic,fi2);
      end
      fic = fic+360;
    end

    if (((fi1<=fic)&&(fi2>=fic)) || ((fi1>=fic)&&(fi2<=fic)))   % it's here!
      semafor = 1;
      if verbose >0
        fprintf('It is here (c)! %f %f %f\n',fi1,fic,fi2);
      end
    else
      if verbose>0
        fprintf('(c)%f %f %f\n',fi1,fic,fi2);
      end
    end

  end

  i=i+1;
end
%
i=i-1;

if semafor == 0
  fprintf('\n\n*** Error: no angle detection. fic = %f x = %f, y = %f \n\n',fic, Point2DX,Point2DY);
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
   if verbose>0
     fprintf('\n The point (%f,%f) is in.\n',Point2DX,Point2DY);
   end
   IsIn = 1;
else
   if verbose>0
     fprintf('\n The point (%f,%f) is out.\n',Point2DX,Point2DY);
   end
   IsIn = 0;
end
%
if verbose>1
  figure (2,'position',[100,200,600,600])   % figure(1, 'position',[startx,starty,width,height]);

  plot(Iris2DX,Iris2DY,'xk');
  hold on;
  plot(Iris2DCX,Iris2DCY,'ok');
  plot(Iris2DX(i),Iris2DY(i),'or');
  plot(Iris2DX(i+1),Iris2DY(i+1),'or');

  if IsIn
  plot(Point2DX,Point2DY,'*b');
  else
  plot(Point2DX,Point2DY,'om');
  end
  %

  plot([Iris2DCX,Point2DX],[Iris2DCY,Point2DY],'-b');
  plot([Iris2DX(1),Iris2DX(end-1)],[Iris2DY(1),Iris2DY(end-1)],'og');
  hold off;

  grid;
  xlabel('X (previously Y)');
  ylabel('Y (previously Z)');
end

end
