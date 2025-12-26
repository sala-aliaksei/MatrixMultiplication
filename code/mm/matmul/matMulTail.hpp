#pragma once
#include <mm/core/Matrix.hpp>

void matMulSimd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMulSimdTails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

#include "mm/core/kernels.hpp"
#include "mm/core/reorderMatrix.hpp"

constexpr int N_LOG_DIM = 25;

template<int Nr, int Kr, int Kc, int... TailSize>
static inline void handleItail(double*       a_buf,
                               const double* a,
                               const double* packed_b,
                               double*       c,
                               int           M,
                               int           N,
                               int           K,
                               int           dNc,
                               int           i_tail_size)
{
    // TODO: Add multithreading

    int i_ofs = 0;

    (...,
     (
       [&]
       {
           constexpr int Mrr = TailSize;
           if (i_tail_size >= Mrr)
           {
               if (N < N_LOG_DIM)
               {
                   std::cout << "-----------------    COMPUTE ITAIL dNc with Mrr = " << Mrr
                             << "  -----------------\n";
               }
               int dMc = i_tail_size - i_tail_size % Mrr;
               reorderColOrderMatrixTail<Mrr, Kr>(a + K * i_ofs, K, a_buf, dMc, Kc);

               // TODO:[Critical] What if Nc%Nr!=0 ??? We need to handle tail here as well
               for (int j = 0; j < dNc; j += Nr)
               {
                   // if (N < N_LOG_DIM)
                   // {
                   //     std::cout << "-----------------     COMPUTE KERNEL IN I
                   //     TAIL for j=" << j
                   //               << "    -----------------\n";
                   // }

                   int  idx    = 0;
                   auto i_tail = i_tail_size;
                   while (i_tail >= Mrr)
                   {
                       kernels::cpp_packed_kernel<Nr, Mrr, Kc>(
                         &a_buf[idx * Kc], &packed_b[Kc * j], &c[(idx + i_ofs) * N + j], N);

                       idx += Mrr;
                       i_tail -= Mrr;
                   }
               }

               i_ofs += i_tail_size - i_tail_size % Mrr;
               i_tail_size %= Mrr;
           }
       }()));
}

template<int Nr, int Kr, int Nc, int Kc, int... TailSize>
static inline void handleItail(double*       a_buf,
                               const double* a,
                               const double* packed_b,
                               double*       c,
                               int           M,
                               int           N,
                               int           K,
                               int           i_tail_size)
{
    // TODO: Add multithreading

    int i_ofs = 0;

    (...,
     (
       [&]
       {
           constexpr int Mrr = TailSize;
           if (i_tail_size >= Mrr)
           {

               if (N < N_LOG_DIM)
               {
                   std::cout << "-----------------    COMPUTE ITAIL Nc   -----------------\n";
               }
               // std::cout << "handle i_tail_size = " << i_tail_size << std::endl;
               int dMc = i_tail_size - i_tail_size % Mrr;
               reorderColOrderMatrixTail<Mrr, Kr>(a + K * i_ofs, K, a_buf, dMc, Kc);

               // TODO: What if Nc%Nr!=0 ???
               for (int j = 0; j < Nc; j += Nr)
               {
                   int  idx    = 0;
                   auto i_tail = i_tail_size;
                   while (i_tail >= Mrr)
                   {
                       kernels::cpp_packed_kernel<Nr, Mrr, Kc>(
                         &a_buf[idx * Kc], &packed_b[Kc * j], &c[(i_ofs + idx) * N + j], N);

                       idx += Mrr;
                       i_tail -= Mrr;
                   }
               }

               i_ofs += i_tail_size - i_tail_size % Mrr;
               i_tail_size %= Mrr;
           }
       }()));
}

template<int Mr, int Nr, int Kr, int Mc, int... TailSize>
static inline void handleKtail(double*       a_buf,
                               double*       b_buf,
                               const double* a,
                               const double* b,
                               double*       c,
                               int           M,
                               int           N,
                               int           K,
                               int           dNc,
                               int           k_tail_size)
{
    // TODO: Add multithreading
    int kofs = 0;
    (...,
     (
       [&]
       {
           constexpr int Kcc = TailSize;
           if (k_tail_size >= Kcc)
           {

               if (N < N_LOG_DIM)
               {
                   std::cout << "-----------------    COMPUTE KTAIL    -----------------\n";
                   std::cout << "k_tail_size = " << k_tail_size << std::endl;
                   std::cout << "Kcc = " << Kcc << std::endl;
                   std::cout << "dNc = " << dNc << std::endl;
                   std::cout << "dNc%Nr = " << dNc % Nr << std::endl;
                   std::cout << "Mc%Mr = " << Mc % Mr << std::endl;
                   std::cout << "kofs = " << kofs << std::endl;
               }

               int dKc = k_tail_size - k_tail_size % Kcc;
               int kdx = kofs;

               for (int k_block = 0; k_block < dKc; k_block += Kcc)
               {
                   reorderRowMajorMatrix<Kr, Nr>(b + N * (k_block + kdx), N, b_buf, Kcc, dNc);

                   int dMc   = M % Mc;
                   int ilast = M - dMc;
                   for (int i_block = 0; i_block < ilast; i_block += Mc)
                   {
                       reorderColOrderMatrix<Mc, Kcc, Mr, Kr>(
                         a + K * i_block + (k_block + kdx), K, a_buf);

                       if (N < N_LOG_DIM)
                       {

                           std::cout
                             << "-----------------     COMPUTE KERNEL IN K TAIL for i_block="
                             << i_block << ", k_block = " << k_block << "  -----------------\n";
                       }
                       for (int j = 0; j < dNc; j += Nr)
                       {
                           const double* Bc1 = b_buf + Kcc * j;
                           for (int i = 0; i < Mc; i += Mr)
                           {
                               double*       Cc0 = c + N * (i_block + i) + j;
                               const double* Ac0 = a_buf + Kcc * i;

                               // TODO: deduce args from span?
                               kernels::cpp_packed_kernel<Nr, Mr, Kcc>(Ac0, Bc1, Cc0, N);
                           }
                       }
                   }

                   const double* Ac1 = a + (k_block + kdx) + K * ilast;
                   double*       Cc1 = c + N * ilast;

                   handleItail<Nr, Kr, Kcc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dNc, dMc);
               }

               kofs += dKc;
               k_tail_size %= Kcc;
           }
       }()));
}

template<int Mr, int Nr, int Kr, int Mc, int Nc, int... TailSize>
static inline void handleKtail(double*       a_buf,
                               double*       b_buf,
                               const double* a,
                               const double* b,
                               double*       c,
                               int           M,
                               int           N,
                               int           K,
                               int           k_tail_size)
{
    // TODO: Add multithreading
    int kofs = 0;
    (...,
     (
       [&]
       {
           constexpr int Kcc = TailSize;
           if (k_tail_size >= Kcc)
           {
               if (N < N_LOG_DIM)
               {
                   std::cout << "-----------------    COMPUTE KTAIL Nc   -----------------\n";
               }

               int dKc = k_tail_size - k_tail_size % Kcc;
               int kdx = kofs;

               for (int k_block = 0; k_block < dKc; k_block += Kcc)
               {
                   reorderRowMajorMatrix<Kcc, Nc, Kr, Nr>(b + N * (k_block + kdx), N, b_buf);

                   int dMc   = M % Mc;
                   int ilast = M - dMc;
                   for (int i_block = 0; i_block < ilast; i_block += Mc)
                   {
                       reorderColOrderMatrix<Mc, Kcc, Mr, Kr>(
                         a + K * i_block + k_block + kdx, K, a_buf);

                       for (int j = 0; j < Nc; j += Nr)
                       {
                           const double* Bc1 = b_buf + Kcc * j;
                           for (int i = 0; i < Mc; i += Mr)
                           {
                               double*       Cc0 = c + N * (i_block + i) + j;
                               const double* Ac0 = a_buf + Kcc * i;

                               // TODO: deduce args from span?
                               kernels::cpp_packed_kernel<Nr, Mr, Kcc>(Ac0, Bc1, Cc0, N);
                           }
                       }
                   }

                   // What if we don't have Mr rows anymore and tail is 1, (new Mr == 1)?

                   const double* Ac1 = a + k_block + kdx + K * ilast;
                   double*       Cc1 = c + N * ilast;

                   handleItail<Nr, Kr, Nc, Kcc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dMc);
               }

               kofs += dKc;
               k_tail_size %= Kcc;
           }
       }()));
}

// TODO: force inline?
template<int Mr, int Kr, int Mc, int Kc, int... TailSize>
static inline void handleJtail(double*       buf,
                               const double* ma,
                               const double* mb,
                               double*       mc,
                               int           M,
                               int           K,
                               int           N,
                               int           j_tail_size)
{

    // TODO: Add multithreading
    int j_ofs = 0;
    (...,
     (
       [&]
       {
           constexpr int Nrr = TailSize;
           if (j_tail_size >= Nrr)
           {
               // dNc % Nrr == 0 always
               int dNc = j_tail_size - j_tail_size % Nrr;

               if (N < N_LOG_DIM)
               {
                   std::cout << "-----------------    COMPUTE JTAIL    -----------------\n";
                   std::cout << "j_tail_size = " << j_tail_size << std::endl;
                   std::cout << "Nrr = " << Nrr << std::endl;
                   std::cout << "dNc = " << dNc << std::endl;
                   std::cout << "j_ofs = " << j_ofs << std::endl;
               }

               double* a_buf = buf;
               double* b_buf = a_buf + Mc * Kc;

               int dKc   = K % Kc;
               int klast = K - dKc;
               for (int k_block = 0; k_block < klast; k_block += Kc)
               {
                   int i_tail_size = M % Mc;
                   int ilast       = M - i_tail_size;

                   for (int i_block = 0; i_block < ilast; i_block += Mc)
                   {

                       reorderColOrderMatrix<Mc, Kc, Mr, Kr>(ma + K * i_block + k_block, K, a_buf);

                       if (N < N_LOG_DIM)
                       {
                           //  std::cout << "ma\n";
                           //  printArr(ma + K * i_block + k_block, Mc, Kc,
                           //  K); std::cout << "a_buf\n"; printArr(a_buf,
                           //  Mc, Kc);
                       }

                       int j_tail = j_tail_size;
                       int jjdx   = j_ofs;

                       while (j_tail >= Nrr)
                       {
                           reorderRowMajorMatrix<Kr, Nrr>(
                             mb + N * k_block + jjdx, N, b_buf, Kc, dNc);

                           if (N < N_LOG_DIM)
                           {
                               //  std::cout << "mb\n";
                               //  printArr(mb + N * k_block + jjdx,
                               //  Kc, Nrr, N);
                               //  std::cout << "b_buf\n";
                               //  printArr(b_buf, Kc, Nrr);

                               std::cout
                                 << "-----------------     COMPUTE KERNEL IN J TAIL for i_block="
                                 << i_block << ", jjdx= " << jjdx << ", k_block = " << k_block
                                 << "  -----------------\n";
                           }

                           // TODO: What if Mc%Mr != 0 ? It can't happen here
                           for (int i = 0; i < Mc; i += Mr)
                           {
                               double*       Cc0 = mc + N * (i_block + i) + jjdx;
                               const double* Ac0 = a_buf + Kc * i;

                               kernels::cpp_packed_kernel<Nrr, Mr, Kc>(Ac0, b_buf, Cc0, N);
                           }

                           j_tail -= Nrr;
                           jjdx += Nrr;
                       }
                   }

                   const double* Ac1 = ma + k_block + K * ilast;
                   const double* Bc1 = mb + N * k_block + j_ofs;
                   double*       Cc1 = mc + N * ilast + j_ofs;

                   reorderRowMajorMatrix<Kr, Nrr>(Bc1, N, b_buf, Kc, dNc);

                   handleItail<Nrr, Kr, Kc, 4, 3, 2, 1>(
                     a_buf, Ac1, b_buf, Cc1, M, N, K, dNc, i_tail_size);
               }

               // TODO: Need to handle K tail; Kc == 80, Ncc is not compile time
               // TODO: Choose probel block sizes for Kcc
               handleKtail<Mr, Nrr, Kr, Mc, 20, 10, 4, 2, 1>(
                 a_buf, b_buf, ma + klast, mb + N * klast + j_ofs, mc + j_ofs, M, N, K, dNc, dKc);

               j_ofs += dNc;
               j_tail_size %= Nrr;
           }
       }()));
}
