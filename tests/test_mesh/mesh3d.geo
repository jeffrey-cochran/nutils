// This geo file defines the [0,2]x[0,1]x[0,1] domain required by
// test_mesh.gmsh, equal to mesh2d.geo but periodically extruded in
// z-direction. To regenerate the msh files:
//
// for o in 1 2; do for v in 2 4; do gmsh -format msh$v -3 -order $o mesh3d.geo -o mesh3d_p${o}_v${v}.msh; done; done

p000 = newp; Point(p000) = {0,0,0};
p001 = newp; Point(p001) = {0,0,1};
p010 = newp; Point(p010) = {0,1,0};
p011 = newp; Point(p011) = {0,1,1};
p100 = newp; Point(p100) = {1,0,0};
p101 = newp; Point(p101) = {1,0,1};
p110 = newp; Point(p110) = {1,1,0};
p111 = newp; Point(p111) = {1,1,1};
p200 = newp; Point(p200) = {2,0,0};
p201 = newp; Point(p201) = {2,0,1};
p210 = newp; Point(p210) = {2,1,0};
p211 = newp; Point(p211) = {2,1,1};
l00X = newl; Line(l00X) = {p000,p001};
l01X = newl; Line(l01X) = {p010,p011};
l0X0 = newl; Line(l0X0) = {p000,p010};
l0X1 = newl; Line(l0X1) = {p001,p011};
l10X = newl; Line(l10X) = {p100,p101};
l11X = newl; Line(l11X) = {p110,p111};
l1X0 = newl; Line(l1X0) = {p100,p110};
l1X1 = newl; Line(l1X1) = {p101,p111};
l20X = newl; Line(l20X) = {p200,p201};
l21X = newl; Line(l21X) = {p210,p211};
l2X0 = newl; Line(l2X0) = {p200,p210};
l2X1 = newl; Line(l2X1) = {p201,p211};
lL00 = newl; Line(lL00) = {p000,p100};
lL01 = newl; Line(lL01) = {p001,p101};
lL10 = newl; Line(lL10) = {p010,p110};
lL11 = newl; Line(lL11) = {p011,p111};
lR00 = newl; Line(lR00) = {p100,p200};
lR01 = newl; Line(lR01) = {p101,p201};
lR10 = newl; Line(lR10) = {p110,p210};
lR11 = newl; Line(lR11) = {p111,p211};
ll0XX = newll; Line Loop(ll0XX) = {l00X,l0X1,-l01X,-l0X0};
ll1XX = newll; Line Loop(ll1XX) = {l1X0,l11X,-l1X1,-l10X};
ll2XX = newll; Line Loop(ll2XX) = {l2X0,l21X,-l2X1,-l20X};
llL0X = newll; Line Loop(llL0X) = {lL00,l10X,-lL01,-l00X};
llL1X = newll; Line Loop(llL1X) = {l01X,lL11,-l11X,-lL10};
llLX0 = newll; Line Loop(llLX0) = {lL00,l1X0,-lL10,-l0X0};
llLX1 = newll; Line Loop(llLX1) = {lL01,l1X1,-lL11,-l0X1};
llR0X = newll; Line Loop(llR0X) = {lR00,l20X,-lR01,-l10X};
llR1X = newll; Line Loop(llR1X) = {l11X,lR11,-l21X,-lR10};
llRX0 = newll; Line Loop(llRX0) = {lR00,l2X0,-lR10,-l1X0};
llRX1 = newll; Line Loop(llRX1) = {lR01,l2X1,-lR11,-l1X1};
s0XX = news; Plane Surface(s0XX) = {ll0XX};
s1XX = news; Plane Surface(s1XX) = {ll1XX};
s2XX = news; Plane Surface(s2XX) = {ll2XX};
sL0X = news; Plane Surface(sL0X) = {llL0X};
sL1X = news; Plane Surface(sL1X) = {llL1X};
sLX0 = news; Plane Surface(sLX0) = {llLX0};
sLX1 = news; Plane Surface(sLX1) = {llLX1};
sR0X = news; Plane Surface(sR0X) = {llR0X};
sR1X = news; Plane Surface(sR1X) = {llR1X};
sRX0 = news; Plane Surface(sRX0) = {llRX0};
sRX1 = news; Plane Surface(sRX1) = {llRX1};
slL = newsl; Surface Loop(slL) = {sL1X,s0XX,sL0X,sLX0,s1XX,sLX1};
slR = newsl; Surface Loop(slR) = {sR1X,s1XX,sR0X,sRX0,s2XX,sRX1};
vL = newv; Volume(vL) = {slL};
vR = newv; Volume(vR) = {slR};
Physical Point("midpoint") = {p100};
Physical Surface("neumann") = {sL0X,sR0X};
Physical Surface("dirichlet") = {s2XX,sR1X,sL1X,s0XX};
Physical Surface("extra") = {sL0X,s0XX};
Physical Surface("iface") = {s1XX};
Physical Volume("left") = {vL};
Physical Volume("right") = {vR};
Periodic Surface {sLX0} = {sLX1} Translate{0,0,-1};
Periodic Surface {sRX0} = {sRX1} Translate{0,0,-1};
