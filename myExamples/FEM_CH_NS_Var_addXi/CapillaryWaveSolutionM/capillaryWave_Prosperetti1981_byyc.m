%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%| Script by Dr. Fabian Denner, f.denner09@imperial.ac.uk,    |%
%|   Dept. Mechanical Engineering, Imperial College London    |%
%|============================================================|%
%| Script to calculate the dispersion of a single capillary   |%
%| wave based on the analytical initial-value solution of     |%
%| Prosperetti (Phys. Fluids 24, 1981, pp. 1217-1223) for     |%
%| two-fluid cases with equal kinematic viscosity and         |%
%| one-fluid cases (setting density and viscosity of the      |%
%| upper fluid to 0).                                         |%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; 
clear;
close all;

%--------------------------------------------------------------
%   Input data.
%--------------------------------------------------------------

% Density upper fluid [kg/m^3]
rhoU = 1.0;
% Viscosity upper fluid [Pa s]
muU = 0.01;

% Density lower fluid [kg/m^3]
rhoL = 1.0;    
% Viscosity lower fluid [Pa s]
% muL = 1.0e-3;       

% Surface tension [N/m]
sigma = 1.;  

% gravity
gravity = 1.0;

% Wavelength [m]
l = 1;    
% Wavenumber [1/m]
k = 2*pi/l;     
% Time steps per period [s]
dtpp = 200;
% Output file name
filename = './capillaryWave';

%--------------------------------------------------------------
%   Calculation of constants for a cosine wave 
%   with a0 = amplitude and no initial velocity (u0 = 0)
%--------------------------------------------------------------

% Density ratio
beta = rhoL * rhoU / ((rhoL + rhoU)^2);
% Kinematic viscosity
v    = muU / rhoU;                      
% Square of inviscid angular freq.
wSq  = sigma * k^3 / (rhoL + rhoU) + gravity*k*(rhoL-rhoU)/(rhoL + rhoU);     
% Initial amplitude
a0   = 0.01;                            
% Initial velocity of center point
u0   = 0;        
% Characteristic timescale
% tau = 1/sqrt(wSq);
tau = 3;
% Time step (different per case!)
% dt = 1.0e-5;   
dt = tau/dtpp;

% Dimensionless viscosity
epsil = v*k^2/sqrt(wSq);

%--------------------------------------------------------------
%   Calculation of roots z1, z2, z3 and z4.
%--------------------------------------------------------------

p1 = 1;
p2 = -4*beta*sqrt(k^2*v);
p3 = 2*(1-6*beta)*k^2*v;
p4 = 4*(1-3*beta)*(k^2*v)^(3/2);
p5 = (1-4*beta)*v^2*k^4 + wSq;

p = [p1 p2 p3 p4 p5];

% Vector with the four roots z1, z2, z3 and z4
z = roots(p);       

%--------------------------------------------------------------
%   Calculation of interface height h.
%--------------------------------------------------------------

t = 0:dt:tau;

part0 = 4*(1-4*beta)*v^2*k^4/(8*(1-4*beta)*v^2*k^4 + wSq) * a0 * ...
        erfc(sqrt(v*k^2*t));

Z1 = (z(2)-z(1))*(z(3)-z(1))*(z(4)-z(1));
part1 = z(1)/Z1 * (wSq*a0/(z(1)^2 - v*k^2) - u0) * exp((z(1)^2-v*k^2)*t) .* ...
        erfc(real(z(1))*sqrt(t));

Z2 = (z(1)-z(2))*(z(3)-z(2))*(z(4)-z(2));
part2 = z(2)/Z2 * (wSq*a0/(z(2)^2 - v*k^2) - u0) * exp((z(2)^2-v*k^2)*t) .* ...
        erfc(real(z(2))*sqrt(t));

Z3 = (z(1)-z(3))*(z(2)-z(3))*(z(4)-z(3));
part3 = z(3)/Z3 * (wSq*a0/(z(3)^2 - v*k^2) - u0) * exp((z(3)^2-v*k^2)*t) .* ...
        erfc(real(z(3))*sqrt(t));

Z4 = (z(1)-z(4))*(z(2)-z(4))*(z(3)-z(4));
part4 = z(4)/Z4 * (wSq*a0/(z(4)^2 - v*k^2) - u0) * exp((z(4)^2-v*k^2)*t) .* ...
        erfc(real(z(4))*sqrt(t));

h = part0 + part1 + part2 + part3 + part4;

%--------------------------------------------------------------
%   Output
%--------------------------------------------------------------
t_h = [t;h];
plot(t,h)
axis([0 tau -a0 a0])
y_val=get(gca,'YTick');    %为了获得y轴句柄
y_str=num2str(y_val');     %为了将数字转换为字符数组
set(gca,'YTickLabel',y_str);     %显示

hold on
grid on
% mat=[t;h];
% csvwrite(filename,transpose(mat))
disp('end of the file')
