// Adapted from MFEM v4.8 examples/ex16.cpp, commit
// a01719101027383954b69af1777dc828bf795d62 (BSD-3-Clause).
// See MFEM-LICENSE and MFEM-NOTICE in this directory.
// Additions: interchangeable GPU SpMV, timing, convergence checks and output.
//                                MFEM Example 16
//
// Compile with: make ex16
//
// Sample runs:  ex16
//               ex16 -m ../data/inline-tri.mesh
//               ex16 -m ../data/disc-nurbs.mesh -tf 2
//               ex16 -s 21 -a 0.0 -k 1.0
//               ex16 -s 22 -a 1.0 -k 0.0
//               ex16 -s 23 -a 0.5 -k 0.5 -o 4
//               ex16 -s 4 -dt 1.0e-4 -tf 4.0e-2 -vs 40
//               ex16 -m ../data/fichera-q2.mesh
//               ex16 -m ../data/fichera-mixed.mesh
//               ex16 -m ../data/escher.mesh
//               ex16 -m ../data/beam-tet.mesh -tf 10 -dt 0.1
//               ex16 -m ../data/amr-quad.mesh -o 4 -r 0
//               ex16 -m ../data/amr-hex.mesh -o 2 -r 0
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. In this example,
//               the diffusion operator is linearized by evaluating with the
//               lagged solution from the previous timestep, so there is only
//               a linear solve.
//
//               We recommend viewing examples 2, 9 and 10 before viewing this
//               example.

#include "spmvOperator.hpp"
#include <iomanip>
#include <memory>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
using atspmv_app::SpmvOperator;
using atspmv_app::Jacobi;
using atspmv_app::Sync;
using atspmv_app::Clock;
using atspmv_app::Seconds;
static atspmv_app::Statistics stats;

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
   FiniteElementSpace &fespace;
   Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

   BilinearForm *M;
   BilinearForm *K;

   SparseMatrix Mmat, Kmat;
   SparseMatrix *T; // T = M + dt K
   real_t current_dt;
    SpmvOperator M_op{stats}, K_op{stats}, T_op{stats};
    bool coefficients_initialized = false;

   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   Jacobi M_prec;  // Preconditioner for the mass matrix M

   CGSolver T_solver; // Implicit solver for T = M + dt K
   Jacobi T_prec;  // Preconditioner for the implicit solver

   real_t alpha, kappa;

   mutable Vector z; // auxiliary vector

public:
   ConductionOperator(FiniteElementSpace &f, real_t alpha, real_t kappa,
                      const Vector &u);

   void Mult(const Vector &u, Vector &du_dt) const override;
   /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
       This is the only requirement for high-order SDIRK implicit integration.*/
   void ImplicitSolve(const real_t dt, const Vector &u, Vector &k) override;

   /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
   void SetParameters(const Vector &u);

    std::pair<double, double> MassEnergy(const Vector &u) const;

   ~ConductionOperator() override;
};

real_t InitialTemperature(const Vector &x);

int main(int argc, char *argv[])
{ try {
   auto total_start = Clock::now();
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 2;
   int order = 2;

   int ode_solver_type = 23;  // SDIRK33Solver
   real_t t_final = 0.5;
   real_t dt = 1.0e-2;
   real_t alpha = 1.0e-2;
   real_t kappa = 0.5;

   bool visualization = false;
   const char *backend = "cusparse";
   const char *solution_file = "heat-final.bin";
   bool visit = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  ODESolver::Types.c_str());
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&alpha, "-a", "--alpha",
                  "Alpha coefficient.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "Kappa coefficient offset.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&backend, "-b", "--backend", "GPU SpMV: mfem, cusparse, atbsr, mixed, tuned.");
   args.AddOption(&solution_file, "-sol", "--solution", "Final true-dof FP64 binary output.");
   args.AddOption(&stats.reuse_values, "-reuse", "--reuse-values", "-rebuild", "--rebuild-values",
                  "Reuse tile structure when CSR offsets and columns match exactly.");
   args.AddOption(&stats.gpu_assembly, "-fa", "--full-assembly", "-legacy", "--legacy-assembly",
                  "Use MFEM's device full assembly instead of the original host assembly.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   stats.backend = backend;
   Device device("cuda");
   device.Print();
   Sync();

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
   auto mesh_owner = std::make_unique<Mesh>(mesh_file, 1, 1);
   Mesh *mesh = mesh_owner.get();
   int dim = mesh->Dimension();

   // 3. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   unique_ptr<ODESolver> ode_solver = ODESolver::Select(ode_solver_type);

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   // 5. Define the vector finite element space representing the current and the
   //    initial temperature, u_ref.
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(mesh, &fe_coll);

   int fe_size = fespace.GetTrueVSize();
   cout << "Number of temperature unknowns: " << fe_size << endl;

   GridFunction u_gf(&fespace);

   // 6. Set the initial conditions for u. All boundaries are considered
   //    natural.
   FunctionCoefficient u_0(InitialTemperature);
   u_gf.ProjectCoefficient(u_0);
   Vector u;
   u_gf.GetTrueDofs(u);

   // 7. Initialize the conduction operator and the visualization.
   ConductionOperator oper(fespace, alpha, kappa, u);
   const auto initial_invariants = oper.MassEnergy(u);

   u_gf.SetFromTrueDofs(u);
   if (visualization || visit)
   {
      ofstream omesh("ex16.mesh");
      omesh.precision(precision);
      mesh->Print(omesh);
      ofstream osol("ex16-init.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   VisItDataCollection visit_dc("Example16", mesh);
   visit_dc.RegisterField("temperature", &u_gf);
   if (visit)
   {
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   socketstream sout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      sout.open(vishost, visport);
      if (!sout)
      {
         cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << endl;
         visualization = false;
         cout << "GLVis visualization disabled.\n";
      }
      else
      {
         sout.precision(precision);
         sout << "solution\n" << *mesh << u_gf;
         sout << "pause\n";
         sout << flush;
         cout << "GLVis visualization paused."
              << " Press space (in the GLVis window) to resume it.\n";
      }
   }

   // 8. Perform time-integration (looping over the time iterations, ti, with a
   //    time-step dt).
   ode_solver->Init(oper);
   real_t t = 0.0;

   Sync();
   double initialization_seconds = Seconds(total_start);
   auto evolution_start = Clock::now();
   bool last_step = false;
   int steps = 0;
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final - dt/2)
      {
         last_step = true;
      }

      ode_solver->Step(u, t, dt);
      ++steps;

      if (last_step || (ti % vis_steps) == 0)
      {
         cout << "step " << ti << ", t = " << t << endl;

         u_gf.SetFromTrueDofs(u);
         if (visualization)
         {
            sout << "solution\n" << *mesh << u_gf << flush;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
      oper.SetParameters(u);
   }

   Sync();
   double evolution_seconds = Seconds(evolution_start);
   double total_seconds = Seconds(total_start);
   const auto final_invariants = oper.MassEnergy(u);
   const double mass_error = std::abs(final_invariants.first-initial_invariants.first)
                             / std::max(1.0, std::abs(initial_invariants.first));
   if (mass_error > 1e-6 || final_invariants.second > initial_invariants.second*(1+1e-6))
      throw std::runtime_error("heat mass conservation or energy decay failed");
   const double *host_u = u.HostRead();
   long double sum = 0, norm = 0;
   for (int i=0; i<u.Size(); ++i) {
      if (!std::isfinite(host_u[i])) throw std::runtime_error("nonfinite temperature");
      sum += host_u[i]; norm += (long double)host_u[i]*host_u[i];
   }
   ofstream solution(solution_file, ios::binary);
   solution.write(reinterpret_cast<const char*>(host_u), sizeof(double)*u.Size());
   if (!solution) throw std::runtime_error("failed to save solution");
   cout << std::setprecision(17) << "APP_RESULT {\"backend\":\"" << backend
        << "\",\"dofs\":" << u.Size() << ",\"steps\":" << steps
        << ",\"nnz\":" << stats.max_nnz << ",\"initialization_s\":" << initialization_seconds
        << ",\"evolution_s\":" << evolution_seconds << ",\"total_s\":" << total_seconds
        << ",\"plan_s\":" << stats.plan_seconds << ",\"plans\":" << stats.plans
        << ",\"value_updates\":" << stats.updates
        << ",\"linear_s\":" << stats.linear_seconds
        << ",\"parameter_s\":" << stats.parameter_seconds
        << ",\"implicit_setup_s\":" << stats.implicit_setup_seconds
        << ",\"spmv_calls\":" << stats.multiplies << ",\"linear_solves\":" << stats.solves
        << ",\"cg_iterations\":" << stats.cg_iterations << ",\"sum\":" << double(sum)
        << ",\"l2\":" << sqrt(double(norm))
        << ",\"mass_relative_error\":" << mass_error
        << ",\"initial_energy\":" << initial_invariants.second
        << ",\"final_energy\":" << final_invariants.second << "}" << endl;
   // 9. Save the final solution. This output can be viewed later using GLVis:
   //    "glvis -m ex16.mesh -g ex16-final.gf".
   if (visualization || visit)
   {
      ofstream osol("ex16-final.gf");
      osol.precision(precision);
      u_gf.Save(osol);
   }

   // 10. Free the used memory.


   return 0;
} catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}

ConductionOperator::ConductionOperator(FiniteElementSpace &f, real_t al,
                                       real_t kap, const Vector &u)
   : TimeDependentOperator(f.GetTrueVSize(), (real_t) 0.0), fespace(f),
     M(NULL), K(NULL), T(NULL), current_dt(0.0), z(height)
{
   const real_t rel_tol = 1e-8;

   M = new BilinearForm(&fespace);
   if (stats.gpu_assembly) {
      M->SetAssemblyLevel(AssemblyLevel::FULL);
      M->EnableSparseMatrixSorting(true);
   }
   M->AddDomainIntegrator(new MassIntegrator());
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(rel_tol);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(30);
   M_solver.SetPrintLevel(0);
   M_solver.SetPreconditioner(M_prec);
   M_op.SetMatrix(Mmat);
   M_solver.SetOperator(M_op);

   alpha = al;
   kappa = kap;

   T_solver.iterative_mode = false;
   T_solver.SetRelTol(rel_tol);
   T_solver.SetAbsTol(0.0);
   T_solver.SetMaxIter(2000);
   T_solver.SetPrintLevel(0);
   T_solver.SetPreconditioner(T_prec);

   SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
   auto solve_start = Clock::now();
   // Compute:
   //    du_dt = M^{-1}*-Ku
   // for du_dt, where K is linearized by using u from the previous timestep
   K_op.Mult(u, z);
   z.Neg(); // z = -z
   M_solver.Mult(z, du_dt);
   if (!M_solver.GetConverged()) throw std::runtime_error("mass CG did not converge");
   stats.cg_iterations += M_solver.GetNumIterations(); ++stats.solves;
   Sync(); stats.linear_seconds += Seconds(solve_start);
}

void ConductionOperator::ImplicitSolve(const real_t dt,
                                       const Vector &u, Vector &du_dt)
{
   // Solve the equation:
   //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
   // for du_dt, where K is linearized by using u from the previous timestep
   if (!T)
   {
      auto setup_start = Clock::now();
      T = Add(1.0, Mmat, dt, Kmat);
      current_dt = dt;
      T_op.SetMatrix(*T);
      T_solver.SetOperator(T_op);
      Sync(); stats.implicit_setup_seconds += Seconds(setup_start);
   }
   MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
   auto solve_start = Clock::now();
   K_op.Mult(u, z);
   z.Neg();
   T_solver.Mult(z, du_dt);
   if (!T_solver.GetConverged()) throw std::runtime_error("implicit CG did not converge");
   stats.cg_iterations += T_solver.GetNumIterations(); ++stats.solves;
   Sync(); stats.linear_seconds += Seconds(solve_start);
}

void ConductionOperator::SetParameters(const Vector &u)
{
   // Constant material coefficients need no reassembly; common to every backend.
   if (alpha == 0.0 && coefficients_initialized) return;
   auto parameter_start = Clock::now();
   GridFunction u_alpha_gf(&fespace);
   u_alpha_gf.SetFromTrueDofs(u);
   for (int i = 0; i < u_alpha_gf.Size(); i++)
   {
      u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
   }

   delete K;
   K = new BilinearForm(&fespace);
   if (stats.gpu_assembly) {
      K->SetAssemblyLevel(AssemblyLevel::FULL);
      K->EnableSparseMatrixSorting(true);
   }

   GridFunctionCoefficient u_coeff(&u_alpha_gf);

   K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);
   K_op.SetMatrix(Kmat);
   coefficients_initialized = true;
   delete T;
   T = NULL; // re-compute T on the next ImplicitSolve
   Sync(); stats.parameter_seconds += Seconds(parameter_start);
}

ConductionOperator::~ConductionOperator()
{
   delete T;
   delete M;
   delete K;
}

std::pair<double, double> ConductionOperator::MassEnergy(const Vector &u) const
{
   const int *ptr=Mmat.HostReadI(), *col=Mmat.HostReadJ();
   const double *val=Mmat.HostReadData(), *x=u.HostRead();
   long double mass=0, energy=0;
   for (int row=0; row<Mmat.Height(); ++row)
   {
      long double mx=0;
      for (int p=ptr[row]; p<ptr[row+1]; ++p) mx+=(long double)val[p]*x[col[p]];
      mass+=mx; energy+=(long double)x[row]*mx;
   }
   return {double(mass),double(energy)};
}

real_t InitialTemperature(const Vector &x)
{
   if (x.Norml2() < 0.5)
   {
      return 2.0;
   }
   else
   {
      return 1.0;
   }
}
