import math

# {{{ param pipe  diameters
r_head = 22.0
r_top = 18.0
r_seat = 14.0
r_down = 19.0
r_bb = 22.0
r_cstay = 14.0
r_sstay = 10.0
r_sstay_s = 12.0
r_brace = 8.0
tria_a = 17.0
tria_b = 15.0
tria_r = 5.0
# }}} param pipe diameters
# {{{ wanddicken
s_head = 3.0
s_top = 2.0
s_seat = 3.0
s_down = 2.0
s_bb = 4.0
s_cstay = 2.5
s_sstay = 3.0
s_brace = 1.8
s_tria = 10.0
# }}} wanddicken
# {{{ material
def nu(E,G):
    return float(E)/(2*G)-1

# almg5
alu_E = 71000.0
alu_G = 25900.0
alu_RHO = 2.64e-6
alu_Rm = 290.0
alu_Rp02 = 150.0
alu_NU = nu(alu_E,alu_G)
alu_plasticity = [alu_Rp02/alu_E,alu_Rp02,alu_Rm/alu_E,alu_Rm]
# }}}


# vim: set filetype=python fdm=marker ts=4 sw=4 tw=79 :
