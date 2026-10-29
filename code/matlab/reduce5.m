function [xxR,yyR,VR] = reduce5 (xx,yy,V)

x = size(xx,1);
y = size(xx,2);

xxR = xx(3:5:end-2,3:5:end-2);
yyR = yy(3:5:end-2,3:5:end-2);

VR = xxR.*0;

ix_ = 1;
for ix = 3:5:x-2

  iy_ = 1;
  for iy = 3:5:y-2

    Vc = V(ix-2:ix+2,iy-2:iy+2);
    VR(ix_,iy_) = mean(mean(Vc));

    iy_ = iy_+1;
  end

  ix_ = ix_+1;

end
%

end
