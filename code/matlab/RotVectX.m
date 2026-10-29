function vr = RotVectX(v, fi_d)
% function rotates vector v around versor vers by angle fi
% consistient version
%
% v - input vector (3,1)
% vers - versor for rotation (around) (3,1)
% fi - rotation angle (1)  - in degrees
%
% https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
%

vers = [1; 0; 0];

fi=fi_d./180.*pi;     % in radians

vers=vers./norm(vers); % to be sure

vr=vers.*dot(vers,v)+cos(fi).*cross(cross(vers,v),vers)+sin(fi).*cross(vers,v);

end
