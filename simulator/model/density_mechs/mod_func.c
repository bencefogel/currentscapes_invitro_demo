#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _car_new_reg();
extern void _exp2synNMDA_reg();
extern void _kad_reg();
extern void _kap_reg();
extern void _kdr_reg();
extern void _kslow_new_reg();
extern void _nadend_reg();
extern void _nax_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," car_new.mod");
fprintf(stderr," exp2synNMDA.mod");
fprintf(stderr," kad.mod");
fprintf(stderr," kap.mod");
fprintf(stderr," kdr.mod");
fprintf(stderr," kslow_new.mod");
fprintf(stderr," nadend.mod");
fprintf(stderr," nax.mod");
fprintf(stderr, "\n");
    }
_car_new_reg();
_exp2synNMDA_reg();
_kad_reg();
_kap_reg();
_kdr_reg();
_kslow_new_reg();
_nadend_reg();
_nax_reg();
}
