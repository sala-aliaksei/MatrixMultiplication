#include "matMulBlis.hpp"

#include "include/haswell/blis.h"

void matmulBlis(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    int m = A.row();
    int n = B.col();
    int k = A.col();

    num_t dt_s = BLIS_DOUBLE;
    num_t dt_d = BLIS_DOUBLE;

    inc_t  rs = 0;
    inc_t  cs = 0;
    obj_t  a, b, c;
    obj_t* alpha;
    obj_t* beta;

    //    bli_obj_create(dt_d, m, n, rs, cs, c);
    //    bli_obj_create(dt_s, m, k, rs, cs, a);
    //    bli_obj_create(dt_s, k, n, rs, cs, b);

    //    bli_obj_set_comp_prec(BLIS_DOUBLE_PREC, &c);
}
